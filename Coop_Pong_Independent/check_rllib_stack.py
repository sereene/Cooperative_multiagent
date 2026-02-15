import os
import numpy as np
import matplotlib
matplotlib.use('Agg') # 화면 없는 서버 대응
import matplotlib.pyplot as plt
import ray
from ray.rllib.models import ModelCatalog
from ray.rllib.algorithms.dqn import DQNConfig
import gymnasium as gym

# 사용자 정의 모듈 임포트
from env_utils import env_creator

def visualize_rllib_stack():
    print("Ray 초기화 중...")
    if ray.is_initialized():
        ray.shutdown()
    ray.init()

    # 1. 환경 생성 (Raw 상태)
    env = env_creator()
    
    # [주의] MultiAgentEnv는 기본적으로 .observation_space가 없을 수 있습니다.
    # 안전하게 첫 번째 에이전트의 공간을 가져옵니다.
    agent_id = env.agents[0]
    obs_space = env.observation_space(agent_id) if hasattr(env, "observation_space") and callable(env.observation_space) else env.observation_space
    
    # 만약 위 방법이 실패하면 PettingZoo 방식 사용
    if obs_space is None:
        obs_space = env.par_env.observation_space(env.agents[0])

    print(f"\n[Environment] Raw Observation Shape: {obs_space.shape}")

    # 2. RLlib Preprocessor 가져오기 (핵심 수정 부분)
    model_config = {
        "custom_model": "custom_cnn",
        "framestack": True,      # 프레임 스택 켜기 (4장)
        "grayscale": True,       # 흑백 변환 켜기
        # "grayscale": False,    # 컬러를 원하시면 False로 변경
    }

    # [수정] get_preprocessor(env) -> get_preprocessor_for_space(obs_space)
    preprocessor = ModelCatalog.get_preprocessor_for_space(obs_space, model_config)
    
    print(f"[RLlib Preprocessor] Type: {type(preprocessor)}")
    print(f"[RLlib Preprocessor] Output Shape: {preprocessor.shape}")
    
    # 3. 데이터 흘려보내기 (Warm-up)
    print("\n게임을 10 스텝 진행하며 스택을 쌓습니다...")
    
    obs, _ = env.reset()
    
    # reset 후 첫 관측값 전처리
    # (paddle_0가 없을 수도 있으니 안전하게 확인)
    target_agent = "paddle_0"
    if target_agent not in obs:
        target_agent = env.agents[0]

    stacked_obs = preprocessor.transform(obs[target_agent])

    for _ in range(10):
        # 랜덤 행동
        actions = {agent: env.action_space.sample() for agent in env.agents}
        obs, _, _, _, _ = env.step(actions)
        
        if target_agent in obs:
            # [중요] RLlib은 step마다 transform을 호출하여 스택을 갱신합니다.
            stacked_obs = preprocessor.transform(obs[target_agent])
        
        # 에피소드 끝나면 리셋
        if not env.agents:
            obs, _ = env.reset()
            preprocessor.reset() 

    # 4. 시각화 (이미지 저장)
    C = stacked_obs.shape[-1]
    
    print(f"\n[Result] Final Stacked Shape: {stacked_obs.shape}")

    plt.figure(figsize=(16, 6))
    
    # Case A: Grayscale=True (채널 4개: 흑백 프레임 4장)
    if C == 4:
        print(">> 감지됨: Grayscale Stack (4 Frames)")
        for i in range(4):
            plt.subplot(1, 4, i+1)
            img = stacked_obs[:, :, i] 
            plt.imshow(img, cmap='gray', vmin=0, vmax=1) 
            plt.title(f"Frame -{3-i}")
            plt.axis('off')

    # Case B: Grayscale=False (채널 12개: RGB 프레임 4장)
    elif C == 12:
        print(">> 감지됨: RGB Stack (4 Frames)")
        for i in range(4):
            plt.subplot(1, 4, i+1)
            start = i * 3
            end = (i + 1) * 3
            img = stacked_obs[:, :, start:end]
            
            if img.max() <= 1.0:
                plt.imshow(img)
            else:
                plt.imshow(img.astype(np.uint8))
                
            plt.title(f"Frame -{3-i}")
            plt.axis('off')
            
    else:
        print(f">> 경고: 예상치 못한 채널 수 ({C})입니다. 첫 채널만 출력합니다.")
        plt.imshow(stacked_obs[:, :, 0])

    save_path = "rllib_internal_stack_check.png"
    plt.tight_layout()
    plt.savefig(save_path)
    
    print(f"\n[완료] RLlib 내부 스택 이미지가 '{save_path}'로 저장되었습니다.")
    
    env.close()
    ray.shutdown()

if __name__ == "__main__":
    visualize_rllib_stack()