import torch
import torch.nn as nn
import numpy as np
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.recurrent_net import RecurrentNetwork
from ray.rllib.policy.rnn_sequencing import add_time_dimension

class CustomCNNGRU(RecurrentNetwork, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        # RecurrentNetwork 초기화
        RecurrentNetwork.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        custom_config = model_config.get("custom_model_config", {})
        # 논문에서 Hidden Dimension은 64
        self.hidden_dim = custom_config.get("fc_size", 64)

        # 관측 공간의 원본 형태 저장
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
            self.cnn_out_dim = self.conv_layers(dummy_input).numel()

        # RNN (GRU) 레이어
        self.fc1 = nn.Linear(self.cnn_out_dim, self.hidden_dim)
        self.gru = nn.GRU(self.hidden_dim, self.hidden_dim, batch_first=True)

        # Output Layers
        self.policy_head = nn.Linear(self.hidden_dim, num_outputs)
        self.value_head = nn.Linear(self.hidden_dim, 1)
        self._value_out = None

    def get_initial_state(self):
        # 초기 Hidden State 반환 [Batch, Hidden_Dim]
        return [self.fc1.weight.new(1, self.hidden_dim).zero_()]

    def forward(self, input_dict, state, seq_lens):
        # 1. Image Processing (CNN)
        x = input_dict["obs"].float()
        x = x / 255.0

        # [입력 형태 복구] 
        # RLLib가 입력을 Flatten해서 주는 경우 ([Batch, Feature]), 이미지 형태로 복구
        if x.dim() == 2:
            x = x.reshape(-1, *self.obs_shape)

        # 채널 차원이 없는 경우 추가 (Batch, H, W) -> (Batch, H, W, 1)
        if x.dim() == 3: 
            x = x.unsqueeze(-1) 
        
        # (Batch, H, W, C) -> (Batch, C, H, W) 로 변경
        x = x.permute(0, 3, 1, 2) 

        x = self.conv_layers(x)
        x = x.reshape(x.size(0), -1) # Flatten [B*T, CNN_Out]
        
        # 2. FC Layer before RNN
        x = torch.relu(self.fc1(x))  # [B*T, Hidden]

        # 3. RNN Processing
        # seq_lens가 None인 경우 (단일 스텝 처리)와 있는 경우 (배치 처리) 분기
        if seq_lens is not None:
            inputs = add_time_dimension(
                x, max_seq_len=seq_lens.max(), framework="torch"
            )
        else:
            # seq_lens가 없으면 Time Dimension=1로 가정하고 차원 추가
            inputs = x.unsqueeze(1)
        
        output, new_state = self.forward_rnn(inputs, state, seq_lens)
        
        # 4. Flatten Output [B*T, Hidden]
        output = torch.reshape(output, [-1, self.hidden_dim])
        
        # 5. Heads
        logits = self.policy_head(output)
        self._value_out = self.value_head(output).squeeze(-1)
        
        return logits, new_state

    def forward_rnn(self, inputs, state, seq_lens):
        # inputs: [Batch, Time, Hidden_Dim]
        # state: List of [Batch, Hidden_Dim]
        
        # GRU 실행
        # state[0] shape: [Batch, Hidden] -> GRU requires [Layers, Batch, Hidden]
        
        # [수정됨] .contiguous() 추가! (RuntimeError: rnn: hx is not contiguous 해결)
        h_0 = state[0].unsqueeze(0).contiguous()
        
        self.gru.flatten_parameters()
        output, h_n = self.gru(inputs, h_0)
        
        # output: [Batch, Time, Hidden]
        # h_n: [Layers, Batch, Hidden] -> [Batch, Hidden]
        return output, [h_n.squeeze(0)]

    def value_function(self):
        return self._value_out