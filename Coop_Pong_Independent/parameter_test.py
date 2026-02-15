from RLlib_DQN_Independent import env_creator
from models import CustomCNN
import gymnasium as gym
import numpy as np

if __name__ == "__main__":
    # 1. 환경 생성 및 Obs/Action Space 추출
    print("Creating environment to fetch observation space...")
    tmp_env = env_creator({})
    obs_space = tmp_env.observation_space
    act_space = tmp_env.action_space
    
    # PettingZoo 등은 MultiAgentEnv라 obs_space가 딕셔너리일 수 있음. 
    # 하지만 RLlib Wrapper를 거치면 개별 에이전트의 Space가 잡히는 경우가 많음.
    # 안전하게 첫 번째 에이전트의 Space를 가져옵니다.
    if hasattr(obs_space, "original_space"):
         obs_space = obs_space.original_space
         
    # 만약 obs_space가 Dict라면 하나만 꺼냄 (모든 에이전트 동일 가정)
    if isinstance(obs_space, gym.spaces.Dict):
        # 보통 RLlib은 각 에이전트의 space를 줍니다. 
        # 여기선 tmp_env가 ParallelPettingZooEnv(RLlib wrapper)이므로
        # observation_space는 1개 에이전트의 space입니다.
        pass
        
    print(f"Detected Observation Shape: {obs_space.shape}")
    print(f"Detected Action Space: {act_space}")

    # 2. 모델 인스턴스화
    # Action Space가 Discrete인 경우 n을 가져옴
    num_outputs = act_space.n if hasattr(act_space, 'n') else 5 

    model = CustomCNN(
        obs_space=obs_space,
        action_space=act_space,
        num_outputs=num_outputs,
        model_config={},
        name="my_cnn"
    )

    # 3. 파라미터 계산
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    model_size_mb = total_params * 4 / (1024 * 1024)  # float32 = 4 bytes

    print("\n" + "="*40)
    print(f" Model Parameter Statistics")
    print("="*40)
    print(f"Input Shape Used       : {obs_space.shape}")
    print(f"Action Outputs         : {num_outputs}")
    print("-" * 40)
    print(f"Trainable Parameters   : {trainable_params:,}")
    print(f"Total Parameters       : {total_params:,}")
    print(f"Estimated Size (MB)    : {model_size_mb:.2f} MB")
    print("="*40)

    tmp_env.close()