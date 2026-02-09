import numpy as np
import torch
import torch.nn as nn
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.annotations import override

class MeltingPotModel(TorchModelV2, nn.Module):
    """
    LSTM 없이 FrameStack을 사용하는 가벼운 CNN 모델
    - 파라미터 수 계산 기능 추가
    - CNN 출력 크기 자동 계산 기능 추가 (stride 변경 대응)
    """
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        in_channels = obs_space.shape[2] 

        # CNN Layers 
        self.conv_layers = nn.Sequential(
            # [88,88,12] -> [11,11,16]
            nn.Conv2d(in_channels, 16, kernel_size=8, stride=8),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=1), 
            nn.ReLU(),
            nn.Flatten()
        )
        
        # CNN 출력 크기 계산
        with torch.no_grad():
            # (Batch=1, C, H, W) 더미 입력 생성
            dummy_input = torch.zeros(1, in_channels, 88, 88) 
            cnn_out = self.conv_layers(dummy_input)
            flatten_size = cnn_out.numel()

        # MLP Layers
        self.fc1 = nn.Linear(flatten_size, 128)
        self.fc2 = nn.Linear(128, 128)

        # heads
        self.policy_head = nn.Linear(128, num_outputs)
        self.value_head = nn.Linear(128, 1)
        
        self._features = None

        # ----------------------------------------------------------------
        # [요청하신 기능] 학습에 참여하는 활성화된 파라미터 수 계산 및 출력
        # ----------------------------------------------------------------
        
        # 1) CNN 파라미터 수
        cnn_params = sum(p.numel() for p in self.conv_layers.parameters() if p.requires_grad)
        
        # 2) 전체 모델 파라미터 수 (CNN + MLP + Heads)
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        print("="*50)
        print(f"🤖 [Model Info] Trainable Parameters Count")
        print(f"   1. After CNN Layers : {cnn_params:,} parameters")
        print(f"   2. Final Total Model: {total_params:,} parameters")
        print("="*50)

    @override(TorchModelV2)
    def forward(self, input_dict, state, seq_lens):
        # 1. 입력 가져오기
        x = input_dict["obs"]
        
        # 정규화
        x = x.float() / 255.0
        
        # 2. 차원 변환: [B, H, W, C] -> [B, C, H, W]
        x = x.permute(0, 3, 1, 2)

        # 3. CNN 통과
        x = self.conv_layers(x)
        
        # 4. MLP 통과
        x = torch.relu(self.fc1(x))
        self._features = torch.relu(self.fc2(x))
        
        # 5. Output
        logits = self.policy_head(self._features)
        
        return logits, []

    @override(TorchModelV2)
    def value_function(self):
        assert self._features is not None, "must call forward() first"
        return self.value_head(self._features).reshape(-1)