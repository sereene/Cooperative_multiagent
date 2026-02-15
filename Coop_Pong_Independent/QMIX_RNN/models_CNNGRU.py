import torch
import torch.nn as nn
import numpy as np
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.recurrent_net import RecurrentNetwork
from ray.rllib.utils.torch_utils import FLOAT_MIN

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

        # CNN 레이어 (TimeDistributed처럼 동작하게 될 것임)
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

        # FC Layer before RNN
        self.fc1 = nn.Linear(self.cnn_out_dim, self.hidden_dim)

        # RNN (GRU) 레이어
        # batch_first=True: 입력이 (Batch, Time, Features) 형태임
        self.gru = nn.GRU(self.hidden_dim, self.hidden_dim, batch_first=True)

        # Output Layers
        self.policy_head = nn.Linear(self.hidden_dim, num_outputs)
        self.value_head = nn.Linear(self.hidden_dim, 1)
        self._value_out = None

    def get_initial_state(self):
        # 초기 Hidden State 반환 [Batch, Hidden_Dim]
        # GRU는 hidden state 하나만 필요 (h_0)
        # 배치 사이즈는 실행 시 결정되므로 여기서는 1개 샘플에 대한 초기값만 정의
        return [self.fc1.weight.new(1, self.hidden_dim).zero_()]

    def forward(self, input_dict, state, seq_lens):
        """
        RLLib의 RecurrentNetwork.forward는 기본적으로 입력을 [Batch * Time, ...] 형태로 평탄화해서 줍니다.
        하지만 RNN 처리를 위해서는 [Batch, Time, ...] 형태가 필요합니다.
        """
        x = input_dict["obs"].float()
        x = x / 255.0

        # 1. CNN Processing (TimeDistributed 효과)
        # 입력 x는 [Batch * Time, H, W, C] 형태임
        if x.dim() == 3: # 그레이스케일이거나 차원이 하나 빠진 경우
            x = x.unsqueeze(-1)
        
        # [Batch * Time, H, W, C] -> [Batch * Time, C, H, W]
        x = x.permute(0, 3, 1, 2)
        
        # CNN 통과
        x = self.conv_layers(x)
        x = x.reshape(x.size(0), -1) # Flatten [Batch * Time, CNN_Out]
        
        # FC Layer
        x = torch.relu(self.fc1(x))  # [Batch * Time, Hidden]

        # 2. RNN Processing (시퀀스 구조 복원)
        # RLLib 유틸리티 대신 직접 차원 변환을 수행하여 명확하게 처리
        
        if seq_lens is not None:
            # 가변 길이 시퀀스 처리
            max_seq_len = x.shape[0] // len(seq_lens)
            batch_size = len(seq_lens)
            
            # [Batch * Time, Hidden] -> [Batch, Time, Hidden]
            # 주의: RLLib이 데이터를 패딩해서 주므로 reshape 가능
            rnn_input = x.reshape(batch_size, max_seq_len, self.hidden_dim)
            
            # Hidden State 준비: [Batch, Hidden] -> [1, Batch, Hidden] (GRU 요구사항)
            h_0 = state[0].unsqueeze(0)
            
            # GRU 실행
            # 여기서 pack_padded_sequence를 쓰면 더 좋지만, 
            # RLLib과의 호환성을 위해 전체 시퀀스를 넣고 나중에 마스킹 처리되는 것에 의존
            # (RLLib은 loss 계산 시 seq_lens를 보고 패딩된 부분을 무시함)
            self.gru.flatten_parameters()
            rnn_output, h_n = self.gru(rnn_input, h_0)
            
            # 결과 다시 평탄화: [Batch, Time, Hidden] -> [Batch * Time, Hidden]
            rnn_output = rnn_output.reshape(-1, self.hidden_dim)
            
            # 다음 State 반환: [Batch, Hidden]
            new_state = [h_n.squeeze(0)]
            
        else:
            # 시퀀스 길이가 없는 경우 (단일 스텝, 배치 크기 1 등)
            # [Batch, Hidden] -> [Batch, 1, Hidden]
            rnn_input = x.unsqueeze(1)
            h_0 = state[0].unsqueeze(0)
            
            self.gru.flatten_parameters()
            rnn_output, h_n = self.gru(rnn_input, h_0)
            
            rnn_output = rnn_output.reshape(-1, self.hidden_dim)
            new_state = [h_n.squeeze(0)]

        # 3. Output Heads
        logits = self.policy_head(rnn_output)
        self._value_out = self.value_head(rnn_output).squeeze(-1)

        return logits, new_state

    def value_function(self):
        return self._value_out