import gc
import os
import numpy as np
import ray
from ray import tune
from ray.rllib.algorithms.dqn import DQNConfig
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.tune.registry import register_env
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
import torch
import torch.nn as nn
from pettingzoo.butterfly import cooperative_pong_v5
import gymnasium as gym
import warnings
import supersuit as ss  
import imageio.v2 as imageio
from ray.air.integrations.wandb import WandbLoggerCallback
from RewardShapingWrapper import RewardShapingWrapper
from MirrorObservationWrapper import MirrorObservationWrapper


# 경고 메시지 무시
warnings.filterwarnings("ignore", category=DeprecationWarning)
os.environ["TUNE_DISABLE_STRICT_METRIC_CHECKING"] = "1"

MAX_CYCLES = 500

class CoopPongCallbacks(DefaultCallbacks):
    def on_episode_end(self, *, worker, base_env, policies, episode, **kwargs):
        length = episode.length
        success = 1.0 if length >= MAX_CYCLES - 1 else 0.0
        episode.custom_metrics["success"] = success

def rollout_and_save_gif(
    *,
    algorithm,
    out_path: str,
    max_cycles: int,
    every_n_steps: int = 4,
    max_frames: int = 200,
    fps: int = 30,
):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    env = cooperative_pong_v5.parallel_env(max_cycles=max_cycles, render_mode="rgb_array")
    env = ss.resize_v1(env, x_size=168, y_size=84)
    # env = ss.color_reduction_v0(env, mode='full')
    # env = ss.frame_stack_v1(env, 4)

    env = RewardShapingWrapper(env)

    env = MirrorObservationWrapper(env)

    frames = []
    try:
        obs, infos = env.reset()
        step_i = 0

        fr0 = env.render()
        if fr0 is not None: frames.append(fr0)

        terminations = {a: False for a in env.possible_agents}
        truncations = {a: False for a in env.possible_agents}

        while True:
            if all(terminations.get(a, False) or truncations.get(a, False) for a in env.possible_agents):
                break

            actions = {}
            for agent_id, agent_obs in obs.items():
                action = algorithm.compute_single_action(agent_obs, policy_id="shared_policy", explore=False)
                actions[agent_id] = action

            obs, rewards, terminations, truncations, infos = env.step(actions)

            if (step_i % every_n_steps) == 0:
                if len(frames) >= max_frames: break
                fr = env.render()
                if fr is not None: frames.append(fr)

            step_i += 1

        if frames:
            imageio.mimsave(out_path, frames, fps=fps)
            print(f"[GIF] saved: {out_path} frames={len(frames)}")
        else:
            print(f"[GIF] skipped (no frames): {out_path}")

    finally:
        try:
            env.close()
            gc.collect()
        except Exception:
            pass

class GifCallbacks(CoopPongCallbacks):
    def __init__(self, out_dir: str, every_n_evals: int = 5, max_cycles: int = 500):
        super().__init__()
        self.out_dir = out_dir
        self.every_n_evals = every_n_evals
        self.max_cycles = max_cycles
        self.eval_count = 0
        os.makedirs(self.out_dir, exist_ok=True)

    def on_train_result(self, *, algorithm, result, **kwargs):
        training_iter = int(result.get("training_iteration", 0))
        if "evaluation" not in result: return

        self.eval_count += 1
        if (self.eval_count % self.every_n_evals) != 0: return

        out_path = os.path.join(self.out_dir, f"eval_{self.eval_count:04d}_iter{training_iter:06d}.gif")
        rollout_and_save_gif(algorithm=algorithm, out_path=out_path, max_cycles=self.max_cycles)

class FixedParallelPettingZooEnv(ParallelPettingZooEnv):
    def __init__(self, env):
        super().__init__(env)
        self.agents = self.par_env.possible_agents
        self._agent_ids = set(self.agents)

    def reset(self, *, seed=None, options=None):
        return self.par_env.reset(seed=seed, options=options)

def env_creator(config=None):
    env = cooperative_pong_v5.parallel_env(max_cycles=MAX_CYCLES, render_mode="rgb_array")

    env = ss.resize_v1(env, x_size=168, y_size=84)
    
    # env = ss.color_reduction_v0(env, mode="full")
    
    # env = ss.frame_stack_v1(env, 4)

    env = RewardShapingWrapper(env)

    env = MirrorObservationWrapper(env)

    return FixedParallelPettingZooEnv(env)

class CustomCNN(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        shape = obs_space.shape
        input_channels = shape[2] if len(shape) == 3 else 1
        input_h, input_w = shape[:2]

        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=input_channels, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        dummy_input = torch.zeros(1, input_channels, input_h, input_w)
        with torch.no_grad():
            self.flatten_dim = self.conv_layers(dummy_input).numel()

        self.fc_net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flatten_dim, 512),
            nn.ReLU()
        )

        self.policy_head = nn.Linear(512, num_outputs)
        self.value_head = nn.Linear(512, 1)
        self._value_out = None

    def forward(self, input_dict, state, seq_lens):
        x = input_dict["obs"].float()
        if x.max() > 10.0: x = x / 255.0
        if x.dim() == 3: x = x.unsqueeze(-1)
        x = x.permute(0, 3, 1, 2)

        x = self.fc_net(self.conv_layers(x))
        logits = self.policy_head(x)
        self._value_out = self.value_head(x).squeeze(-1)
        return logits, state

    def value_function(self):
        return self._value_out

ModelCatalog.register_custom_model("custom_cnn", CustomCNN)

if __name__ == "__main__":
    ray.init()

    env_name = "cooperative_pong_shared_DoubleDQN"
    register_env(env_name, lambda cfg: env_creator(cfg))
    
    tmp_env = env_creator({})
    obs_space = tmp_env.observation_space
    act_space = tmp_env.action_space
    tmp_env.close()

    policies = {"shared_policy": (None, obs_space, act_space, {})}
    def policy_mapping_fn(agent_id, *args, **kwargs): return "shared_policy"

    current_dir = os.getcwd()
    local_log_dir = os.path.join(current_dir, "results")
    experiment_name = "DoubleDQN_CoopPong_SharedPolicy_CNN_RewardShaping_MirrorObs"

    config = (
        DQNConfig()
        .rl_module(_enable_rl_module_api=False)
        .training(_enable_learner_api=False)
        .environment(env=env_name, clip_actions=True, disable_env_checking=True)
        .framework("torch")
        .rollouts(
            num_rollout_workers=4,
            rollout_fragment_length=4, 
            compress_observations=True
        )
        .training(
            model={"custom_model": "custom_cnn"},
            
            # --- Double DQN Specifics ---
            # 핵심: Double Q-Learning 활성화
            double_q=True, 
            
            # 레포지토리는 "Double DQN"이므로 Dueling은 끔
            dueling=False, 
            
            # Atari 기본 설정들
            num_atoms=1,
            v_min=-10.0, v_max=10.0,
            noisy=False,
            
            replay_buffer_config={
                "type": "MultiAgentReplayBuffer",
                "capacity": 1_000_000, # 메모리 부족 시 100,000으로 축소 권장
            },
            
            n_step=1,
            target_network_update_freq=10000, # 1만 스텝마다 타겟 업데이트
            train_batch_size=32,
            
            lr=1e-4, 
            gamma=0.99,
        )
        # 탐험 설정 (Epsilon Greedy)
        # Nature DQN: 1.0 -> 0.1
        # Double DQN Repo: 보통 1.0 -> 0.01 (더 오래, 더 깊게 탐험 후 수렴)
        .exploration(
            exploration_config={
                "type": "EpsilonGreedy",
                "initial_epsilon": 1.0,
                "final_epsilon": 0.01,       # 1%까지 줄임
                "epsilon_timesteps": 1_000_000, # 100만 스텝 동안 감소
            }
        )
        .callbacks(lambda: GifCallbacks(out_dir=os.path.join(local_log_dir, experiment_name, "gifs")))
        .multi_agent(
            policies=policies,
            policy_mapping_fn=policy_mapping_fn,
            policies_to_train=["shared_policy"],
        )
        .evaluation(
            evaluation_interval=50,
            evaluation_num_episodes=10,
            evaluation_config={"explore": False},
        )
        .resources(num_gpus=1 if torch.cuda.is_available() else 0)
    )

    print(f"### Training Logs will be saved at: {local_log_dir} ###")

    tune.run(
        "DQN",
        name=experiment_name,
        stop={"timesteps_total": 10_000_000},
        local_dir=local_log_dir,
        metric="evaluation/custom_metrics/success_mean",
        mode="max",
        keep_checkpoints_num=2,
        checkpoint_score_attr="evaluation/custom_metrics/success_mean",
        checkpoint_freq=100,
        checkpoint_at_end=True,
        config=config.to_dict(),
        callbacks=[
            WandbLoggerCallback(
                project="cooperative_pong_multiagent_shared_policy",
                group="ppo_experiments",
                job_type="training",
                name=experiment_name,
                log_config=True
            )
        ]
    )