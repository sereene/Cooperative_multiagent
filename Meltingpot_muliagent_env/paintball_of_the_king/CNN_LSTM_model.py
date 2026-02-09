import torch
import torch.nn as nn
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.recurrent_net import RecurrentNetwork

class MeltingPotModel(RecurrentNetwork, nn.Module):

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)


        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=8, stride=8),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )
        
        # CNN 출력 크기 계산
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, 88, 88)
            cnn_out_dim = self.conv_layers(dummy_input).numel()

        # MLP (Pre-LSTM)
        self.mlp = nn.Sequential(
            nn.Linear(cnn_out_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )

        # LSTM
        # followed by an LSTM with 128 units
        self.lstm_state_size = 128
        self.lstm = nn.LSTM(128, self.lstm_state_size, batch_first=True)

        # Heads (Policy & Value)
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

        h_state, c_state = state
        
        input_batch_size = inputs.shape[0]
        state_batch_size = h_state.shape[0]
        
        if input_batch_size != state_batch_size:
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
        
        # State Return: [Batch, Hidden] 
        return logits, [h_out.squeeze(0), c_out.squeeze(0)]

    def forward(self, input_dict, state, seq_lens):

        x = input_dict["obs"].float()
        
        if x.max() > 10.0: 
            x = x / 255.0 

        if len(x.shape) == 5:
            B, T, H, W, C = x.shape
            x = x.reshape(B * T, H, W, C)
        elif len(x.shape) == 4:
            B, H, W, C = x.shape
            T = 1
        else:
            raise ValueError(f"Unexpected input shape: {x.shape}")
        
        # PyTorch Conv2D 입력 형식으로 변환: [N, C, H, W]
        x = x.permute(0, 3, 1, 2)
        
        # CNN -> MLP
        x = self.conv_layers(x)
        x = self.mlp(x)
        
        # LSTM 입력을 위해 시퀀스 형태로 복원: [Batch, Time, Features]
        x = x.reshape(B, T, -1)
        
        # RNN Pass
        logits, new_state = self.forward_rnn(x, state, seq_lens)
        
        # Output Flattening
        logits = logits.reshape(B * T, -1)
        self._value_out = self._value_out.reshape(B * T)
        
        return logits, new_state
