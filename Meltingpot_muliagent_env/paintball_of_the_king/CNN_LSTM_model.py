import torch
import torch.nn as nn
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.recurrent_net import RecurrentNetwork

class MeltingPotModel(RecurrentNetwork, nn.Module):
    """
    Melting Pot 2.0 Paper 'CB' Agent Architecture for PPO
    Structure: ConvNet -> MLP(64, 64) -> LSTM(128) -> Actor/Critic Heads
    """
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        # 1. ConvNet
        # "two layers with 16, 32 output channels, kernel shapes 8, 4, and strides 8, 1"
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=8, stride=8),  # [88,88,3] -> [11,11,16]
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=1), # [11,11,16] -> [8,8,32]
            nn.ReLU(),
            nn.Flatten()
        )
        
        # CNN 출력 크기 계산
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, 88, 88)
            cnn_out_dim = self.conv_layers(dummy_input).numel()

        # 2. MLP (Pre-LSTM)
        # "followed by an MLP with two layers with 64 neurons each"
        self.mlp = nn.Sequential(
            nn.Linear(cnn_out_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )

        # 3. LSTM
        # "followed by an LSTM with 128 units"
        self.lstm_state_size = 128
        self.lstm = nn.LSTM(64, self.lstm_state_size, batch_first=True)

        # 4. Heads (Policy & Value)
        self.policy_head = nn.Linear(self.lstm_state_size, num_outputs)
        self.value_head = nn.Linear(self.lstm_state_size, 1)
        self._value_out = None

    def get_initial_state(self):
        return [
            torch.zeros(self.lstm_state_size),
            torch.zeros(self.lstm_state_size),
        ]

    def value_function(self):
        return torch.reshape(self._value_out, [-1])

    def forward_rnn(self, inputs, state, seq_lens):
        # inputs: [Batch, Time, Features]
        # state: List of [Batch, Hidden_Dim] (h, c)
        
        h_state, c_state = state
        
        # [방어 로직] 입력 배치 크기와 Hidden State 배치 크기가 다를 경우 (초기화 시 발생)
        input_batch_size = inputs.shape[0]
        state_batch_size = h_state.shape[0]
        
        if input_batch_size != state_batch_size:
            # 상태를 현재 입력 배치 크기에 맞게 새로 생성 (0으로 초기화)
            device = inputs.device
            h_state = torch.zeros(input_batch_size, self.lstm_state_size, device=device)
            c_state = torch.zeros(input_batch_size, self.lstm_state_size, device=device)
            
        # LSTM 입력 차원: (Num_Layers, Batch, Hidden)
        h_input = h_state.unsqueeze(0)
        c_input = c_state.unsqueeze(0)
        
        # LSTM Pass
        self._features, [h_out, c_out] = self.lstm(inputs, (h_input, c_input))
        
        # Outputs: [Batch, Time, Out]
        logits = self.policy_head(self._features)
        self._value_out = self.value_head(self._features)
        
        # State Return: [Batch, Hidden] (레이어 차원 제거)
        return logits, [h_out.squeeze(0), c_out.squeeze(0)]

    def forward(self, input_dict, state, seq_lens):
        # 1. 입력 데이터 가져오기 및 정규화
        x = input_dict["obs"]["RGB"].float()
        if x.max() > 10.0: 
            x = x / 255.0 

        # 2. 입력 차원 확인 및 변형 (Reshape)
        if len(x.shape) == 5:
            # 학습 중: [Batch, Time, H, W, C]
            B, T, H, W, C = x.shape
            x = x.reshape(B * T, H, W, C)
        elif len(x.shape) == 4:
            # 초기화/단일 스텝: [Batch, H, W, C] -> T=1 가정
            B, H, W, C = x.shape
            T = 1
        else:
            raise ValueError(f"Unexpected input shape: {x.shape}")
        
        # 3. PyTorch Conv2D 입력 형식으로 변환: [N, C, H, W]
        x = x.permute(0, 3, 1, 2)
        
        # 4. CNN -> MLP
        x = self.conv_layers(x)
        x = self.mlp(x)
        
        # 5. LSTM 입력을 위해 시퀀스 형태로 복원: [Batch, Time, Features]
        x = x.reshape(B, T, -1)
        
        # 6. RNN Pass
        logits, new_state = self.forward_rnn(x, state, seq_lens)
        
        # 7. [핵심 수정] Output Flattening: [B, T, Out] -> [B * T, Out]
        # RLlib은 forward 결과가 평탄화된 2차원 텐서여야 합니다.
        logits = logits.reshape(B * T, -1)
        self._value_out = self._value_out.reshape(B * T)
        
        return logits, new_state