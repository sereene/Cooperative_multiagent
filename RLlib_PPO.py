import os
import shutil

import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.tune.registry import register_env
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2

import torch
import torch.nn as nn
from RewardShapingWrapper import RewardShapingWrapper

from pettingzoo.butterfly import cooperative_pong_v5
from ray.air.integrations.wandb import WandbLoggerCallback 


# 평가 지표 누락 시 에러 무시
os.environ["TUNE_DISABLE_STRICT_METRIC_CHECKING"] = "1"

MAX_CYCLES = 256


class FixedParallelPettingZooEnv(ParallelPettingZooEnv):
    def __init__(self, env):
        super().__init__(env)
        self.agents = self.par_env.possible_agents
        self._agent_ids = set(self.agents)

    def reset(self, *, seed=None, options=None):
        obs, info = self.par_env.reset(seed=seed, options=options)
        return obs, info

def env_creator(config=None):

    env = cooperative_pong_v5.parallel_env(
        max_cycles=MAX_CYCLES,
        render_mode="rgb_array"
    )

    # reward shaping wrapper 추가
    env = RewardShapingWrapper(env)
    
    return FixedParallelPettingZooEnv(env)


class CoopPongCallbacks(DefaultCallbacks):
    def on_episode_end(self, *, worker, base_env, policies, episode, **kwargs):
        length = episode.length
        success = 1.0 if length >= MAX_CYCLES - 1 else 0.0
        episode.custom_metrics["success"] = success

# Flatten + MLP (CNN 필터 학습 X)
class FlattenMLP(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        # 1/4 크기로 다운샘플링
        self.pre_process = nn.MaxPool2d(kernel_size=4, stride=4) 
        # 원본 해상 크기: 280x480x3
        input_dim = 70 * 120 * 3

        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, 256),
            nn.ReLU(),
        )

        self.policy_head = nn.Linear(256, num_outputs)
        self.value_head = nn.Linear(256, 1)
        self._value_out = None

    def forward(self, input_dict, state, seq_lens):
        x = input_dict["obs"].float() / 255.0
        x = x.permute(0, 3, 1, 2)

        x = self.pre_process(x)
        x = self.mlp(x)

        logits = self.policy_head(x) # 정책 출력
        self._value_out = self.value_head(x).squeeze(-1) # 가치 함수 출력
        return logits, state

    def value_function(self):
        return self._value_out

ModelCatalog.register_custom_model("flatten_mlp", FlattenMLP)

if __name__ == "__main__":
    ray.init()

    env_name = "cooperative_pong_parallel"
    register_env(env_name, lambda cfg: env_creator(cfg))
    
    tmp_env = env_creator({})
    obs_space = tmp_env.observation_space
    act_space = tmp_env.action_space
    tmp_env.close()

    policies = {
        "policy_left": (None, obs_space, act_space, {}),
        "policy_right": (None, obs_space, act_space, {}),
    }

    def policy_mapping_fn(agent_id, *args, **kwargs):
        return "policy_left" if "0" in agent_id else "policy_right"

    current_dir = os.getcwd()
    local_log_dir = os.path.join(current_dir, "results")

    config = (
        PPOConfig()
        .rl_module(_enable_rl_module_api=False)
        .training(_enable_learner_api=False)
        
        .environment(
            env=env_name, 
            clip_actions=True,
            disable_env_checking=True 
        )
        .framework("torch")
        .rollouts(
            num_rollout_workers=4,
            rollout_fragment_length=256, 
        )
        .training(
            model={
                "custom_model": "flatten_mlp",
            },
            train_batch_size=4 * 256,
            sgd_minibatch_size=256,
            num_sgd_iter=8,
            lr=5e-6,
            gamma=0.99,
            lambda_=0.95,
            use_gae=True,
            vf_loss_coeff=0.5,
            entropy_coeff=0.0,
        )
        .callbacks(CoopPongCallbacks)
        .multi_agent(
            policies=policies,
            policy_mapping_fn=policy_mapping_fn,
        )
        .evaluation(
            evaluation_interval=100,
            evaluation_num_episodes=25,
            evaluation_config={
                "explore": False,
            },
        )
        .resources(
            num_gpus=1 if torch.cuda.is_available() else 0
        )
    )

    print(f"### Training Logs will be saved at: {local_log_dir} ###")

    tune.run(
        "PPO",
        name="PPO_cooperative_pong_v5",
        stop={"timesteps_total": 20_000_000},
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
                project="cooperative_pong_v5",
                group="ppo_experiments",
                job_type="training",
                log_config=True
            )
        ]
    )
