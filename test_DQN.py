import os
import time
import argparse
import warnings
import numpy as np

import ray
from ray.tune.registry import register_env
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
import torch
import torch.nn as nn
import supersuit as ss
from pettingzoo.butterfly import cooperative_pong_v5

# [사용자 정의 Wrapper 임포트]
from RewardShapingWrapper import RewardShapingWrapper
from MirrorObservationWrapper import MirrorObservationWrapper

RENDER = True 
NUM_EPISODES = 5
MAX_CYCLES = 500

# [중요] 체크포인트 경로 수정
CHECKPOINT_PATH = "/home/jsr/project/Cooperative_pong_RL_agent/checkpoint_best"

# ==============================================================================
# 1. 수정된 Wrapper Class (여기가 핵심)
# ==============================================================================
class FixedParallelPettingZooEnv(ParallelPettingZooEnv):
    def __init__(self, env):
        super().__init__(env)
        self.agents = self.par_env.possible_agents
        self._agent_ids = set(self.agents)

    def reset(self, *, seed=None, options=None):
        return self.par_env.reset(seed=seed, options=options)

    # [수정됨] render 호출 시 인자를 받지 않고 내부 par_env.render()를 호출
    def render(self):
        return self.par_env.render()

# ==============================================================================
# 2. Custom Model (이전과 동일)
# ==============================================================================
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

# ==============================================================================
# 3. Environment Creator & Setup
# ==============================================================================
def env_creator(config=None):
    if config is None: config = {}
    
    # 렌더링 모드 설정 ("human" or "rgb_array")
    mode = config.get("render_mode", "rgb_array")
    
    env = cooperative_pong_v5.parallel_env(
        max_cycles=MAX_CYCLES,
        render_mode=mode  # 여기서 모드를 결정합니다.
    )
    
    env = ss.resize_v1(env, x_size=168, y_size=84)
    env = RewardShapingWrapper(env)
    env = MirrorObservationWrapper(env)

    return FixedParallelPettingZooEnv(env)

def policy_mapping_fn(agent_id, *args, **kwargs):
    return "shared_policy"

# ==============================================================================
# 4. Main Test Function
# ==============================================================================
def run_test():
    if ray.is_initialized():
        ray.shutdown()
    ray.init(ignore_reinit_error=True)

    env_name = "cooperative_pong_shared_DoubleDQN"
    register_env(env_name, lambda cfg: env_creator(cfg))

    print(f"Loading checkpoint from: {CHECKPOINT_PATH}")
    try:
        algo = Algorithm.from_checkpoint(CHECKPOINT_PATH)
    except Exception as e:
        print(f"\n[Error] Checkpoint Loading Failed: {e}")
        return

    # 테스트 환경 생성 (화면 표시용 'human')
    test_render_mode = "human" if RENDER else "rgb_array"
    print(f"Creating environment with render_mode='{test_render_mode}'...")
    env = env_creator({"render_mode": test_render_mode})

    for i in range(NUM_EPISODES):
        obs, info = env.reset()
        
        # [중요] 인자 없이 호출
        if RENDER: env.render()
        
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
            
            done = all(terminations.values()) or all(truncations.values())
            total_reward += sum(rewards.values())
            step_count += 1

            if RENDER:
                # [중요] 인자 없이 호출 (par_env.render()로 위임됨)
                env.render()
                time.sleep(0.05)

        print(f"Episode {i+1}: Steps={step_count}, Reward={total_reward:.2f}")

    env.close()
    ray.shutdown()
    print("Test Finished.")

if __name__ == "__main__":
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"Error: Path not found {CHECKPOINT_PATH}")
    else:
        run_test()