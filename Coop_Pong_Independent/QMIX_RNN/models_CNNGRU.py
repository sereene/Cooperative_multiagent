import torch
import torch.nn as nn
import numpy as np
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.recurrent_net import RecurrentNetwork

class CustomCNNGRU(RecurrentNetwork, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        RecurrentNetwork.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        custom_config = model_config.get("custom_model_config", {})
        self.hidden_dim = custom_config.get("fc_size", 128)

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

        # FC Layer before RNN
        self.fc1 = nn.Linear(self.cnn_out_dim, self.hidden_dim)

        # RNN (GRU) 레이어
        self.gru = nn.GRU(self.hidden_dim, self.hidden_dim, batch_first=True)

        # Output Layers
        self.policy_head = nn.Linear(self.hidden_dim, num_outputs)
        self.value_head = nn.Linear(self.hidden_dim, 1)
        self._value_out = None

    def get_initial_state(self):
        # 초기 Hidden State 반환 [Batch, Hidden_Dim]
        return [self.fc1.weight.new(1, self.hidden_dim).zero_()]

    def forward(self, input_dict, state, seq_lens):
        x = input_dict["obs"].float()
        x = x / 255.0

        # 1. 입력 형태 복원 (Flattened -> Image)
        if x.dim() == 2:
            if len(self.obs_shape) == 3:
                x = x.reshape(x.size(0), self.obs_shape[0], self.obs_shape[1], self.obs_shape[2])
            else:
                x = x.reshape(x.size(0), self.obs_shape[0], self.obs_shape[1], 1)

        # 2. CNN Processing
        if x.dim() == 3: 
            x = x.unsqueeze(-1)
        
        x = x.permute(0, 3, 1, 2) 
        
        x = self.conv_layers(x)
        x = x.reshape(x.size(0), -1) 
        
        x = torch.relu(self.fc1(x)) 

        # 3. RNN Processing
        if seq_lens is not None and len(seq_lens) > 0:
            batch_size = seq_lens.shape[0] if isinstance(seq_lens, torch.Tensor) else len(seq_lens)
            max_seq_len = x.shape[0] // batch_size
            
            rnn_input = x.view(batch_size, max_seq_len, self.hidden_dim)
            
            h_0 = state[0].unsqueeze(0).contiguous()
            
            self.gru.flatten_parameters()
            rnn_output, h_n = self.gru(rnn_input, h_0)
            
            rnn_output = rnn_output.view(-1, self.hidden_dim)
            new_state = [h_n.squeeze(0)]
            
        else:
            rnn_input = x.unsqueeze(1)
            h_0 = state[0].unsqueeze(0).contiguous()
            
            self.gru.flatten_parameters()
            rnn_output, h_n = self.gru(rnn_input, h_0)
            
            rnn_output = rnn_output.view(-1, self.hidden_dim)
            new_state = [h_n.squeeze(0)]

        # 4. Output Heads
        logits = self.policy_head(rnn_output)
        self._value_out = self.value_head(rnn_output).squeeze(-1)

        return logits, new_state

    def value_function(self):
        return self._value_out
