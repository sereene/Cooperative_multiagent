import torch
import torch.nn as nn
import numpy as np
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2

class CustomCNN(TorchModelV2, nn.Module):
    # [핵심] 클래스 변수로 플래그 선언 (모든 인스턴스가 이 변수를 공유함)
    _summary_printed = False

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        custom_config = model_config.get("custom_model_config", {})
        fc_dim = custom_config.get("fc_size", 512)

        # 관측 공간의 원본 형태 저장 (복원을 위해 필수)
        self.obs_shape = obs_space.shape
        
        # 입력 채널 및 크기 계산
        if len(self.obs_shape) == 3:
            input_h, input_w, input_channels = self.obs_shape
        else:
            # (H, W)인 경우
            input_h, input_w = self.obs_shape
            input_channels = 1

        # CNN 레이어
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=input_channels, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        # Flatten 차원 계산
        dummy_input = torch.zeros(1, input_channels, input_h, input_w)
        with torch.no_grad():
            self.flatten_dim = self.conv_layers(dummy_input).numel()

        # fc 레이어
        self.fc_net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flatten_dim, fc_dim), 
            nn.ReLU()
        )

        self.policy_head = nn.Linear(fc_dim, num_outputs)
        self.value_head = nn.Linear(fc_dim, 1)
        self._value_out = None

    def forward(self, input_dict, state, seq_lens):
        x = input_dict["obs"].float()
        x = x / 255.0 

        # [FIX] 입력이 평탄화되어(2D) 들어올 경우, 이미지 형태로 복원
        # x.shape 예시: [Batch, 14112] -> [Batch, 84, 168, 1]
        if x.dim() == 2:
            if len(self.obs_shape) == 3:
                x = x.reshape(x.size(0), self.obs_shape[0], self.obs_shape[1], self.obs_shape[2])
            else:
                x = x.reshape(x.size(0), self.obs_shape[0], self.obs_shape[1], 1)

        # [FIX] (Batch, H, W)인 경우 -> (Batch, H, W, 1)로 차원 확장
        if x.dim() == 3: 
            x = x.unsqueeze(-1)
            
        # [Batch, H, W, C] -> [Batch, C, H, W]
        x = x.permute(0, 3, 1, 2)

        x = self.fc_net(self.conv_layers(x))
        logits = self.policy_head(x)
        self._value_out = self.value_head(x).squeeze(-1)
        return logits, state

    def value_function(self):
        return self._value_out