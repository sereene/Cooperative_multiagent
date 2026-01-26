import torch
import numpy as np
import matplotlib
# 화면이 없는 서버 환경에서도 저장 가능하도록 설정
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

# 사용자 정의 모듈 임포트
from env_utils import env_creator
from models import CustomCNN

def save_cnn_feature_maps():
    print("1. 환경 및 모델 준비 중...")
    
    # 환경 생성
    env = env_creator()
    
    # 모델 인스턴스 생성
    # obs_space와 act_space도 RLLib 래퍼에서는 속성으로 접근합니다.
    obs_space = env.observation_space
    act_space = env.action_space
    
    model = CustomCNN(
        obs_space=obs_space,
        action_space=act_space,
        num_outputs=3,
        model_config={},
        name="vis_cnn"
    )
    
    print("2. 데이터 확보 중 (Warm-up 10 steps)...")
    obs, _ = env.reset()
    
    # [수정된 부분]
    # RLLib 래퍼된 환경에서는 env.action_space가 함수가 아니라 객체입니다.
    # 따라서 env.action_space(agent).sample() 대신 env.action_space.sample()을 씁니다.
    for _ in range(10):
        actions = {agent: env.action_space.sample() for agent in env.agents}
        obs, _, _, _, _ = env.step(actions)
        
        # 에피소드가 끝나면 리셋
        if not env.agents: 
            obs, _ = env.reset()

    # paddle_0의 관측값 가져오기
    # 만약 에이전트가 하나도 없다면(에러 방지) 다시 리셋
    if "paddle_0" not in obs:
        obs, _ = env.reset()
        
    agent_obs = obs["paddle_0"] 

    # 3. 모델 입력 형태로 전처리
    input_tensor = torch.from_numpy(agent_obs).float()
    
    # 정규화 (0~255 -> 0~1) - models.py 로직과 동일하게
    if input_tensor.max() > 10.0:
        input_tensor = input_tensor / 255.0
        
    # 차원 변경: (H, W, C) -> (C, H, W) -> (Batch, C, H, W)
    input_tensor = input_tensor.permute(2, 0, 1).unsqueeze(0) 

    print("3. CNN 통과 및 특징 맵 추출...")
    with torch.no_grad():
        feature_maps = model.conv_layers(input_tensor)

    # 4. 시각화 및 저장
    features = feature_maps[0].cpu().numpy()
    num_channels = features.shape[0]
    
    cols = 8
    rows = (num_channels + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(20, 3 * rows))
    fig.suptitle(f"CNN Output Feature Maps\n(Channels: {num_channels}, Shape: {features.shape[1:]})", fontsize=16)
    
    axes = axes.flatten()
    for i in range(len(axes)):
        if i < num_channels:
            axes[i].imshow(features[i], cmap='viridis') 
            axes[i].set_title(f"Channel {i}")
            axes[i].axis('off')
        else:
            axes[i].axis('off')

    plt.tight_layout()
    
    save_filename = "cnn_output_features.png"
    plt.savefig(save_filename)
    print(f"\n[성공] CNN 특징 맵이 '{save_filename}' 파일로 저장되었습니다.")
    
    env.close()

if __name__ == "__main__":
    save_cnn_feature_maps()