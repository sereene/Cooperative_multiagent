import numpy as np
import torch
import torch.nn as nn
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2

class CustomMLP(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        # Vector Observation은 Frame Stack이 적용되면 (27, 4)와 같은 형태가 됩니다.
        # 이를 1차원으로 펴서(Flatten) 입력 크기를 계산합니다.
        input_dim = int(np.prod(obs_space.shape))

        self.fc_net = nn.Sequential(
            nn.Flatten(),  # (Batch, 27, 4) -> (Batch, 108)
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )

        self.policy_head = nn.Linear(128, num_outputs)
        self.value_head = nn.Linear(128, 1)
        self._value_out = None

    def forward(self, input_dict, state, seq_lens):
        x = input_dict["obs"].float()
        
        # 모델 통과
        features = self.fc_net(x)
        logits = self.policy_head(features)
        self._value_out = self.value_head(features).squeeze(-1)
        
        return logits, state

    def value_function(self):
        return self._value_out