import os
import numpy as np
import ray
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.models import ModelCatalog
from ray.rllib.policy.policy import Policy
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
import torch.nn as nn
from pettingzoo.mpe import simple_tag_v3
import supersuit as ss
import warnings
from ray.tune.registry import register_env

from MPE_simple_tag.MPE_tag_PPO_shared import env_creator 

# 경고 무시
warnings.filterwarnings("ignore")

# ======================================================================
# [필수] 학습 때 사용한 클래스들을 그대로 정의해야 합니다.
# (Pickle 로딩 시 이 클래스 정보가 필요합니다)
# ======================================================================

class FlattenMLP(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        input_dim = obs_space.shape[0]
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),       
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(256, num_outputs)
        self.value_head = nn.Linear(256, 1)
        self._value_out = None

    def forward(self, input_dict, state, seq_lens):
        x = input_dict["obs"].float()
        x = self.mlp(x)
        logits = self.policy_head(x)
        self._value_out = self.value_head(x).squeeze(-1)
        return logits, state

    def value_function(self):
        return self._value_out

class EscapeHeuristicPolicy(Policy):
    def __init__(self, observation_space, action_space, config):
        super().__init__(observation_space, action_space, config)

    def compute_actions(self, obs_batch, state_batches, prev_action_batch=None, prev_reward_batch=None, **kwargs):
        batch_size = len(obs_batch)
        actions = [np.random.choice([1, 2, 3, 4]) for _ in range(batch_size)]
        return np.array(actions), [], {}

    def learn_on_batch(self, samples):
        return {}

    def get_weights(self):
        return {}

    def set_weights(self, weights):
        pass

# 모델 등록 (필수)
ModelCatalog.register_custom_model("flatten_mlp", FlattenMLP)

# ======================================================================
# [설정] 체크포인트 경로 입력
# 예: "/home/user/ray_results/PPO_simple_tag.../checkpoint_000100"
# ======================================================================
CHECKPOINT_PATH = "/home/jsr/project/Cooperative_pong_RL_agent/checkpoint_MPE" 
# 위 경로를 본인의 실제 체크포인트 경로로 바꿔주세요!

def run_inference():
    ray.init()
    register_env("simple_tag_ppo_shared", lambda config: env_creator(config))

    print(f"Loading checkpoint from: {CHECKPOINT_PATH}")
    
    # 1. 체크포인트 불러오기
    # Algorithm.from_checkpoint를 쓰면 config를 따로 설정 안 해도 저장된 설정을 가져옵니다.
    try:
        algo = Algorithm.from_checkpoint(CHECKPOINT_PATH)
    except Exception as e:
        print(f"체크포인트 로드 실패: {e}")
        print("경로가 정확한지, 커스텀 클래스가 정의되었는지 확인해주세요.")
        return

    # 2. 테스트용 환경 생성 (render_mode='human'으로 설정하여 화면 출력)
    print("Setting up environment for visualization...")
    env = simple_tag_v3.parallel_env(
        num_good=1, 
        num_adversaries=2, 
        num_obstacles=2, 
        max_cycles=1000, # 테스트니까 좀 길게 봄
        continuous_actions=False,
        render_mode="human" # [중요] 화면에 창을 띄워서 보여줌
    )
    
    # 전처리 (학습과 동일하게 맞춰야 함)
    env = ss.pad_observations_v0(env)
    env = ss.pad_action_space_v0(env)

    # 3. 인퍼런스 루프
    print("Starting simulation... (Press Ctrl+C to stop)")
    
    for episode_num in range(5): # 5판 정도 테스트
        obs, info = env.reset()
        done = False
        total_reward = 0
        step = 0
        
        while not done:
            actions = {}
            for agent_id, agent_obs in obs.items():
                # 정책 매핑: 추격자는 학습된 정책, 도망자는 휴리스틱
                if "adversary" in agent_id:
                    policy_id = "shared_adversary_policy"
                else:
                    policy_id = "agent_policy"

                # 행동 결정 (explore=False로 설정하여 학습된 최적 행동만 선택)
                action = algo.compute_single_action(
                    agent_obs, 
                    policy_id=policy_id, 
                    explore=False
                )
                actions[agent_id] = action
            
            # 환경 진행
            obs, rewards, terminations, truncations, infos = env.step(actions)
            env.render() # 화면 갱신
            
            # 종료 조건 체크
            done = all(terminations.values()) or all(truncations.values())
            
            # 간단한 리워드 출력 (추격자 것만)
            adv_reward = sum([r for a, r in rewards.items() if "adversary" in a])
            if adv_reward > 5: # 충돌 발생 시 출력
                print(f"Step {step}: Collision! (Reward: {adv_reward})")
                
            step += 1
            
        print(f"Episode {episode_num + 1} finished.")

    env.close()
    ray.shutdown()

if __name__ == "__main__":
    run_inference()