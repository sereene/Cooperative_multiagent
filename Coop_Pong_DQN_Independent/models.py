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
        fc_dim = custom_config.get("fc_size", 256)

        shape = obs_space.shape
        input_channels = shape[2] if len(shape) == 3 else 1
        input_h, input_w = shape[:2]

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

        # =================================================================
        # [수정] 클래스 변수를 확인하여 처음 한 번만 출력
        # =================================================================
        if not CustomCNN._summary_printed:
            total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
            
            print(f"\n{'='*40}")
            print(f"[Model Info] {name}")
            print(f"Observation Shape: {shape}")
            print(f"Input Channels   : {input_channels}")
            print(f"Flatten Dim      : {self.flatten_dim}")
            print(f"FC Layer Dim     : {fc_dim}")  # 추가됨
            print(f"Total Parameters : {total_params:,}") 
            print(f"{'='*40}\n")
            
            # 출력 후 플래그를 True로 변경하여 다시 출력되지 않게 함
            CustomCNN._summary_printed = True
        # =================================================================

    def forward(self, input_dict, state, seq_lens):
        x = input_dict["obs"].float()
        x = x / 255.0 

        if x.dim() == 3: x = x.unsqueeze(-1)
        x = x.permute(0, 3, 1, 2)

        x = self.fc_net(self.conv_layers(x))
        logits = self.policy_head(x)
        self._value_out = self.value_head(x).squeeze(-1)
        return logits, state

    def value_function(self):
        return self._value_out
