import os
import time
import argparse
import warnings

import ray
from ray.tune.registry import register_env
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
import torch
import torch.nn as nn
from pettingzoo.butterfly import cooperative_pong_v5


RENDER = True 
NUM_EPISODES = 3

# [중요] 체크포인트 경로 (본인의 경로로 수정)
CHECKPOINT_PATH = "/home/jsr/project/Cooperative_pong_RL_agent/checkpoint2"

MAX_CYCLES = 500


class FixedParallelPettingZooEnv(ParallelPettingZooEnv):
    def __init__(self, env):
        super().__init__(env)
        self.agents = self.par_env.possible_agents
        self._agent_ids = set(self.agents)

    def reset(self, *, seed=None, options=None):
        obs, info = self.par_env.reset(seed=seed, options=options)
        return obs, info

# config 인자를 받아서 렌더링 모드를 결정하도록 수정
def env_creator(config=None):
    if config is None:
        config = {}
    
    # config 딕셔너리에 'render_mode'가 있으면 그걸 따름 (테스트 루프용)
    # 없으면 기본적으로 'rgb_array' 사용 (Agent 로딩용 -> 창 안 뜸)
    mode = config.get("render_mode", "rgb_array")
    
    env = cooperative_pong_v5.parallel_env(
        max_cycles=MAX_CYCLES,
        render_mode=mode
    )
    return FixedParallelPettingZooEnv(env)

# class FlattenMLP(TorchModelV2, nn.Module):
#     def __init__(self, obs_space, action_space, num_outputs, model_config, name):
#         TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
#         nn.Module.__init__(self)
        
#         self.pre_process = nn.AvgPool2d(kernel_size=4, stride=4) 
#         input_dim = 70 * 120 * 3
        
#         self.mlp = nn.Sequential(
#             nn.Flatten(),
#             nn.Linear(input_dim, 128),
#             nn.ReLU(),
#         )
#         self.policy_head = nn.Linear(128, num_outputs)
#         self.value_head = nn.Linear(128, 1)
#         self._value_out = None

#     def forward(self, input_dict, state, seq_lens):
#         x = input_dict["obs"].float() / 255.0
#         x = x.permute(0, 3, 1, 2)
#         x = self.pre_process(x)
#         x = self.mlp(x)
#         logits = self.policy_head(x)
#         self._value_out = self.value_head(x).squeeze(-1)
#         return logits, state

#     def value_function(self):
#         return self._value_out

from RLlib_PPO import FlattenMLP

ModelCatalog.register_custom_model("flatten_mlp", FlattenMLP)

def policy_mapping_fn(agent_id, *args, **kwargs):
    return "policy_left" if "0" in agent_id else "policy_right"

# 실행 로직
def run_test():
    ray.init()

    # config 없이 호출되므로 'rgb_array' 모드 
    # Agent가 내부적으로 환경을 만들 때는 이 설정이 사용됨
    env_name = "cooperative_pong_parallel"
    register_env(env_name, lambda cfg: env_creator(cfg))

    print(f"Loading checkpoint from: {CHECKPOINT_PATH}")
    
    try:
        algo = Algorithm.from_checkpoint(CHECKPOINT_PATH)
    except Exception as e:
        print(f"\n[Error] 체크포인트를 불러올 수 없습니다.\n{e}")
        return

    # 테스트 루프용 환경 생성: 여기서만 명시적으로 "human" 요청
    test_render_mode = "human" if RENDER else "rgb_array"
    env = env_creator({"render_mode": test_render_mode})

    print(f"\nStart Testing for {NUM_EPISODES} episodes...")
    print(f"Render Mode: {test_render_mode}\n")

    for i in range(NUM_EPISODES):
        obs, info = env.reset()
        done = False
        total_reward = 0
        step_count = 0

        while not done:
            actions = {}
            for agent_id, agent_obs in obs.items():
                policy_id = policy_mapping_fn(agent_id)
                action = algo.compute_single_action(
                    observation=agent_obs,
                    policy_id=policy_id,
                    explore=False 
                )
                actions[agent_id] = action

            obs, rewards, terminations, truncations, infos = env.step(actions)
            
            done = len(env.agents) == 0 or all(terminations.values()) or all(truncations.values())
            
            step_reward = sum(rewards.values())
            total_reward += step_reward
            step_count += 1

            if RENDER:
                time.sleep(0.03) 

        print(f"Episode {i+1}: Length={step_count}, Total Reward={total_reward:.2f}")

    env.close()
    ray.shutdown()
    print("\nTest Finished.")

if __name__ == "__main__":
    run_test()