import torch
import torch.nn as nn
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2

#Flatten MLP 모델
class FlattenMLP(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        # 1. 입력 이미지의 형태 파악
        shape = obs_space.shape
        
        # 흑백(2D)인 경우와 3차원인 경우 구분
        if len(shape) == 2:
            input_h, input_w = shape
            input_channels = 1  # 흑백이므로 1
        elif len(shape) == 3:
            input_h, input_w, input_channels = shape
        else:
            raise ValueError(f"Unsupported shape: {shape}")

        # 1/4 크기로 다운샘플링 (Pooling)
        self.pre_process = nn.MaxPool2d(kernel_size=4, stride=4)
        
        # 다운샘플링 후 크기 계산 (280x480 기준)
        # H: 280 // 4 = 70
        # W: 480 // 4 = 120
        final_h = int(input_h // 4)
        final_w = int(input_w // 4)
        
        # 최종 Flatten 차원 계산
        # 70 * 120 * 1 = 8,400
        input_dim = final_h * final_w * input_channels
        
        print(f"DEBUG: Detected Input Channels: {input_channels}") 
        print(f"DEBUG: Model Input Dim initialized as: {input_dim}") # 8400이 찍혀야 정상

        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, 512), 
            nn.ReLU(),
            nn.Linear(512, 256),       
            nn.ReLU(),
        )

        self.policy_head = nn.Linear(256, num_outputs)
        self.value_head = nn.Linear(256, 1)
        self._value_out = None

    def forward(self, input_dict, state, seq_lens):
        x = input_dict["obs"].float() / 255.0

        # [중요] 흑백 이미지 처리 로직 추가
        # ss.color_reduction을 쓰면 (Batch, H, W) 형태로 들어올 수 있음
        # Conv2d나 MaxPool2d는 (Batch, C, H, W)를 기대하므로 차원 추가 필요
        if x.dim() == 3:  # [Batch, H, W]
            x = x.unsqueeze(-1)  # -> [Batch, H, W, 1]

        # [Batch, H, W, C] -> [Batch, C, H, W] 로 순서 변경
        x = x.permute(0, 3, 1, 2)

        x = self.pre_process(x)
        x = self.mlp(x)

        logits = self.policy_head(x)
        self._value_out = self.value_head(x).squeeze(-1)
        return logits, state

    def value_function(self):
        return self._value_out
    

