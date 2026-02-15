import os
import numpy as np
import ray
import torch
import imageio.v2 as imageio
from ray.rllib.policy import Policy
from ray.rllib.models import ModelCatalog

# 기존 사용자 파일 임포트
from MLPmodels import CustomMLP
from FrameStackWrapper import FrameStackWrapper
from env_utils import FixedParallelPettingZooEnv
from pettingzoo.butterfly import knights_archers_zombies_v10

# 모델 등록 (필수)
ModelCatalog.register_custom_model("custom_mlp", CustomMLP)

def mixed_env_creator(render_mode="rgb_array"):
    """기사 1명, 궁수 1명이 등장하는 혼합 환경 생성"""
    env = knights_archers_zombies_v10.parallel_env(
        spawn_rate=50,
        num_archers=1,   # 궁수 1
        num_knights=1,   # 기사 1
        max_arrows=10,
        max_cycles=500,
        vector_state=True,
        render_mode=render_mode
    )
    env = FrameStackWrapper(env, num_stack=3)
    env = FixedParallelPettingZooEnv(env)
    return env

def load_policy_from_checkpoint(checkpoint_path, original_policy_id):
    """
    체크포인트 경로에서 특정 정책을 로드합니다.
    checkpoint_path: Algorithm 체크포인트 폴더 (예: checkpoint_000200)
    original_policy_id: 학습 당시 정책 이름 (예: 'knight_0' or 'default_policy')
    """
    # 정책 체크포인트 경로는 보통 checkpoint_dir/policies/policy_id 형태입니다.
    policy_path = os.path.join(checkpoint_path, "policies", original_policy_id)
    
    if not os.path.exists(policy_path):
        raise FileNotFoundError(f"Policy path not found: {policy_path}")
    
    print(f"Loading policy from: {policy_path}")
    return Policy.from_checkpoint(policy_path)

def run_inference(knight_ckpt, archer_ckpt, output_gif="mixed_play.gif"):
    ray.init(ignore_reinit_error=True)
    
    # 1. 정책 로드
    # 주의: 이전에 학습할 때 사용한 에이전트 ID(정책 ID)를 정확히 입력해야 합니다.
    # 사용자의 train.py를 보면 'knight_0', 'knight_1' 등을 사용했습니다.
    # 궁수 학습시 'archer_0'를 썼다고 가정합니다.
    
    # 기사 정책 로드 (학습된 체크포인트 내의 정책 이름이 'knight_0'라고 가정)
    policy_knight = load_policy_from_checkpoint(knight_ckpt, "knight_0")
    
    # 궁수 정책 로드 (학습된 체크포인트 내의 정책 이름이 'archer_0'라고 가정)
    # 만약 Independent DQN으로 학습했다면 보통 에이전트 이름이 정책 이름입니다.
    policy_archer = load_policy_from_checkpoint(archer_ckpt, "archer_0")

    # 2. 환경 생성
    env = mixed_env_creator()
    obs, infos = env.reset()
    
    frames = []
    
    # 초기 렌더링
    if hasattr(env, "par_env"):
        frames.append(env.par_env.render())
    
    print("Start Simulation...")
    
    total_reward = {"knight_0": 0, "archer_0": 0}
    
    for _ in range(500): # max cycles
        actions = {}
        
        for agent_id, agent_obs in obs.items():
            if "knight" in agent_id:
                # 기사 정책 사용 (Explore=False로 결정적 행동)
                action, _, _ = policy_knight.compute_single_action(agent_obs, explore=False)
            elif "archer" in agent_id:
                # 궁수 정책 사용
                action, _, _ = policy_archer.compute_single_action(agent_obs, explore=False)
            else:
                continue
                
            actions[agent_id] = action
            
        obs, rewards, terminations, truncations, infos = env.step(actions)
        
        # 보상 누적
        for agent, r in rewards.items():
            total_reward[agent] += r

        # 렌더링
        if hasattr(env, "par_env"):
            frames.append(env.par_env.render())
            
        if terminations.get("__all__", False) or truncations.get("__all__", False) or len(obs) == 0:
            break
            
    env.close()
    
    # GIF 저장
    imageio.mimsave(output_gif, frames, fps=30)
    print(f"Saved GIF to {output_gif}")
    print(f"Total Rewards: {total_reward}")
    
    ray.shutdown()

if __name__ == "__main__":
    # [설정] 여기에 실제 체크포인트 경로를 입력하세요.
    # 예: "results/KAZ_Independent.../checkpoint_000200"
    
    KNIGHT_CHECKPOINT_PATH = "/home/jsr/project/Cooperative_pong_RL_agent/KAZ_DQN_Independent/results/KAZ_Independent_DQN_MLP_VectorObs_2Knights/DQN_kaz_independent_DoubleDQN_Vector_5ec4b_00000_0_2026-02-10_03-37-21/checkpoint_000010"
    ARCHER_CHECKPOINT_PATH = "/home/jsr/project/Cooperative_pong_RL_agent/KAZ_DQN_Independent/results/KAZ_Independent_DQN_MLP_VectorObs_noStack_knightRewardShaping/DQN_kaz_independent_DoubleDQN_Vector_b3d3e_00000_0_2026-01-20_04-04-38/checkpoint_000032"
    
    # 궁수 체크포인트가 따로 없다면, 테스트를 위해 기사 체크포인트를 둘 다 넣어도 동작은 합니다.
    run_inference(KNIGHT_CHECKPOINT_PATH, ARCHER_CHECKPOINT_PATH)