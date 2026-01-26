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
# [환경 변경] Entombed Cooperative
from pettingzoo.atari import entombed_cooperative_v3
import gymnasium as gym
import warnings
import supersuit as ss  
import imageio.v2 as imageio
from ray.air.integrations.wandb import WandbLoggerCallback

# 경고 메시지 무시
warnings.filterwarnings("ignore", category=DeprecationWarning)
os.environ["TUNE_DISABLE_STRICT_METRIC_CHECKING"] = "1"

MAX_CYCLES = 2000 

# -------------------------------------------------------------------------
# [추가] Agent ID를 이미지 채널로 추가하는 Wrapper
# -------------------------------------------------------------------------
class AddAgentIdWrapper:
    def __init__(self, env):
        self.env = env
        self.possible_agents = env.possible_agents
        self.metadata = env.metadata
        self.render_mode = env.render_mode
        
        # 기존 Observation Space 가져오기
        old_space = env.observation_space(self.possible_agents[0])
        
        # 채널 1개 추가 (RGB 3 -> RGB+ID 4)
        new_shape = list(old_space.shape)
        new_shape[-1] = 4 
        
        low = np.zeros(new_shape, dtype=old_space.dtype)
        high = np.full(new_shape, 255, dtype=old_space.dtype)
        
        self._new_space = gym.spaces.Box(low=low, high=high, dtype=old_space.dtype)
        self.observation_spaces = {agent: self._new_space for agent in self.possible_agents}

    # PettingZoo API: 함수로 호출됨
    def observation_space(self, agent):
        return self.observation_spaces[agent]

    # PettingZoo API: 함수로 호출됨
    def action_space(self, agent):
        return self.env.action_space(agent)

    def reset(self, seed=None, options=None):
        obs, infos = self.env.reset(seed=seed, options=options)
        return self._add_id_channel(obs), infos

    def step(self, actions):
        obs, rewards, terms, truncs, infos = self.env.step(actions)
        return self._add_id_channel(obs), rewards, terms, truncs, infos

    def render(self):
        return self.env.render()

    def close(self):
        return self.env.close()

    def _add_id_channel(self, obs_dict):
        new_obs = {}
        for agent_id, img in obs_dict.items():
            if "first" in agent_id:
                id_value = 0
            else:
                id_value = 255
            
            h, w, _ = img.shape
            id_channel = np.full((h, w, 1), id_value, dtype=img.dtype)
            
            new_img = np.concatenate([img, id_channel], axis=-1)
            new_obs[agent_id] = new_img
            
        return new_obs
    
    # 그 외 메서드(agents 등)는 원래 env로 패스
    def __getattr__(self, name):
        return getattr(self.env, name)

# -------------------------------------------------------------------------
# Callbacks
# -------------------------------------------------------------------------
class EntombedCallbacks(DefaultCallbacks):
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
    max_frames: int = 300,
    fps: int = 30,
):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # GIF 생성 시에도 동일한 전처리 파이프라인 적용 필수
    env = entombed_cooperative_v3.parallel_env(max_cycles=max_cycles, render_mode="rgb_array")
    env = ss.resize_v1(env, x_size=84, y_size=84)
    env = AddAgentIdWrapper(env) # [중요] ID Wrapper 적용

    frames = []
    try:
        obs, infos = env.reset()
        step_i = 0

        # 초기 렌더링
        fr0 = env.render()
        if fr0 is not None: frames.append(fr0)

        terminations = {a: False for a in env.possible_agents}
        truncations = {a: False for a in env.possible_agents}

        while True:
            if all(terminations.get(a, False) or truncations.get(a, False) for a in env.possible_agents):
                break

            actions = {}
            for agent_id, agent_obs in obs.items():
                # Shared Policy 사용
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

class GifCallbacks(EntombedCallbacks):
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

# -------------------------------------------------------------------------
# Environment Creator
# -------------------------------------------------------------------------
class FixedParallelPettingZooEnv(ParallelPettingZooEnv):
    def __init__(self, env):
        super().__init__(env)
        self.agents = self.par_env.possible_agents
        self._agent_ids = set(self.agents)

    def reset(self, *, seed=None, options=None):
        return self.par_env.reset(seed=seed, options=options)

def env_creator(config=None):
    env = entombed_cooperative_v3.parallel_env(max_cycles=MAX_CYCLES, render_mode="rgb_array")
    
    # 1. Resize (3 channels)
    env = ss.resize_v1(env, x_size=84, y_size=84)
    
    # 2. Reward Shaping

    # 3. Add Agent ID Channel (3 channels -> 4 channels)
    # 반드시 Resize 이후에 적용해야 함 (채널 수가 바뀌므로)
    env = AddAgentIdWrapper(env)

    return FixedParallelPettingZooEnv(env)

# -------------------------------------------------------------------------
# Custom CNN Model
# -------------------------------------------------------------------------
class CustomCNN(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        shape = obs_space.shape
        # shape[2]는 이제 4가 됩니다. (RGB + ID)
        input_channels = shape[2] if len(shape) == 3 else 1
        input_h, input_w = shape[:2]
        
        print(f"[DEBUG] CustomCNN Input Channels: {input_channels} (Expected 4)")

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
        
        # 정규화: 0~255 값을 0~1.0으로 변환
        if x.max() > 10.0: x = x / 255.0
        
        # 차원 보정
        if x.dim() == 3: x = x.unsqueeze(-1)
        
        # (Batch, H, W, C) -> (Batch, C, H, W)
        x = x.permute(0, 3, 1, 2)

        x = self.fc_net(self.conv_layers(x))
        logits = self.policy_head(x)
        self._value_out = self.value_head(x).squeeze(-1)
        return logits, state

    def value_function(self):
        return self._value_out

ModelCatalog.register_custom_model("custom_cnn", CustomCNN)

# -------------------------------------------------------------------------
# Main Training Script
# -------------------------------------------------------------------------
if __name__ == "__main__":
    ray.init()

    # 실험 이름
    env_name = "entombed_shared_with_ID"
    register_env(env_name, lambda cfg: env_creator(cfg))
    
    # Obs Space 확인
    tmp_env = env_creator({})
    obs_space = tmp_env.observation_space
    act_space = tmp_env.action_space
    tmp_env.close()
    
    print(f"Observation Space Shape: {obs_space.shape}") # (84, 168, 4) 여야 함

    # Shared Policy 정의
    policies = {
        "shared_policy": (None, obs_space, act_space, {}),
    }

    def policy_mapping_fn(agent_id, *args, **kwargs): 
        return "shared_policy"

    current_dir = os.getcwd()
    local_log_dir = os.path.join(current_dir, "results")
    experiment_name = "DoubleDQN_Entombed_Shared_IDChannel_CNN_5e-5"

    config = (
        DQNConfig()
        .rl_module(_enable_rl_module_api=False)
        .training(_enable_learner_api=False)
        .environment(env=env_name, clip_actions=True, disable_env_checking=True)
        .framework("torch")
        .rollouts(
            num_rollout_workers=8,
            rollout_fragment_length=4, 
            compress_observations=True
        )
        .training(
            model={"custom_model": "custom_cnn"},
            
            # --- Double DQN Specifics ---
            double_q=True, 
            dueling=True, 
            num_atoms=1,
            v_min=-10.0, v_max=10.0,
            noisy=False,
            
            replay_buffer_config={
                "type": "MultiAgentReplayBuffer",
                "capacity": 50_000, 
            },
            
            n_step=1,
            target_network_update_freq=10000, 
            train_batch_size=32,
            
            lr=5e-5, 
            gamma=0.99,
        )
        .exploration(
            exploration_config={
                "type": "EpsilonGreedy",
                "initial_epsilon": 1.0,
                "final_epsilon": 0.01,       
                "epsilon_timesteps": 4_000_000, 
            }
        )
        .callbacks(lambda: GifCallbacks(
            out_dir=os.path.join(local_log_dir, experiment_name, "gifs"),
            max_cycles=MAX_CYCLES
        ))
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
                project="entombed_cooperative_shared", 
                group="dqn_experiments",
                job_type="training",
                name=experiment_name,
                log_config=True
            )
        ]
    )