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
        # seq_lens가 텐서로 들어올 때를 대비한 안전한 조건문
        if seq_lens is not None and len(seq_lens) > 0:
            # 텐서인 경우 shape[0]으로, 리스트인 경우 len()으로 배치 사이즈 추출
            batch_size = seq_lens.shape[0] if isinstance(seq_lens, torch.Tensor) else len(seq_lens)
            max_seq_len = x.shape[0] // batch_size
            
            # [수정됨] reshape -> view 변경: 메모리상의 시계열 순서가 정확한지 강제 검증
            rnn_input = x.view(batch_size, max_seq_len, self.hidden_dim)
            
            # GRU의 초기 상태 차원 맞추기: (num_layers, batch_size, hidden_dim)
            h_0 = state[0].unsqueeze(0).contiguous()
            
            self.gru.flatten_parameters()
            rnn_output, h_n = self.gru(rnn_input, h_0)
            
            # 다시 RLlib이 기대하는 1차원 배치(B*T) 형태로 복구
            rnn_output = rnn_output.view(-1, self.hidden_dim)
            new_state = [h_n.squeeze(0)]
            
        else:
            # Inference(평가) 환경 등에서 seq_lens가 없을 때의 단일 스텝 처리
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
    

import torch.nn.functional as F
import numpy as np

import torch
import torch.nn as nn
import numpy as np

class CustomImageQMixer(nn.Module):
    def __init__(self, n_agents, state_shape, mixing_embed_dim):
        super(CustomImageQMixer, self).__init__()

        self.n_agents = n_agents
        self.embed_dim = mixing_embed_dim
        
        # RLlib이 주는 Flatten된 원본 State의 길이 (예: 84*84*3*N)
        self.state_dim = int(np.prod(state_shape))

        # ==========================================
        # [추가됨] 1. 이미지 복원을 위한 차원 설정 
        # ==========================================
        # RGB(3채널) 대신 흑백(1채널) 사용 시 아래와 같이 수정
        self.obs_shape = (84, 84, 1) # H, W, C
        
        # 만약 환경에서 (84, 84)처럼 2차원으로만 준다면 아래 로직으로 에러 방지
        if len(self.obs_shape) == 3:
            self.input_h, self.input_w, self.input_channels = self.obs_shape
        else:
            self.input_h, self.input_w = self.obs_shape
            self.input_channels = 1

        # ==========================================
        # [추가됨] 2. 첫 번째 에이전트 화면용 CNN 
        # ==========================================
        self.state_cnn = nn.Sequential(
            nn.Conv2d(self.input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        # CNN을 거친 후의 1D 피처 크기 계산
        dummy = torch.zeros(1, self.input_channels, self.input_h, self.input_w)
        with torch.no_grad():
            cnn_out_dim = self.state_cnn(dummy).numel()

        # CNN 피처를 받아서 하이퍼네트워크에 넘겨줄 최종 임베딩 차원 축소
        self.state_emb_dim = 128
        self.state_fc = nn.Sequential(
            nn.Linear(cnn_out_dim, self.state_emb_dim),
            nn.ReLU()
        )

        # ==========================================
        # 3. 원본 하이퍼네트워크 구조 (입력만 state_dim -> state_emb_dim으로 변경)
        # ==========================================
        self.hyper_w_1 = nn.Linear(self.state_emb_dim, self.embed_dim * self.n_agents)
        self.hyper_w_final = nn.Linear(self.state_emb_dim, self.embed_dim)
        self.hyper_b_1 = nn.Linear(self.state_emb_dim, self.embed_dim)
        
        self.V = nn.Sequential(
            nn.Linear(self.state_emb_dim, self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, 1),
        )

    def forward(self, agent_qs, states):
        """Forward pass for the mixer."""
        bs = agent_qs.size(0)
        
        # 1. RLlib이 주는 1차원 배열을 가져옴
        states = states.reshape(-1, self.state_dim).float()

        # 픽셀값(0~255)이 들어올 경우 정규화 (0~1)
        if states.max() > 1.0:
            states = states / 255.0

        # ==========================================
        # [추가됨] 2. 이미지 차원 복원 및 중복 제거(Slicing)
        # ==========================================
        # [B*T, N, H, W, C] 로 복원
        x = states.view(-1, self.n_agents, self.input_h, self.input_w, self.input_channels)
        
        # 첫 번째 에이전트 화면만 추출 -> [B*T, H, W, C]
        x = x[:, 0, :, :, :]
        
        # PyTorch 형식 [B*T, C, H, W] 로 변환
        x = x.permute(0, 3, 1, 2).contiguous()

        # CNN 인코더 통과
        x = self.state_cnn(x)
        x = x.view(x.size(0), -1)
        state_emb = self.state_fc(x) # 최종 [B*T, state_emb_dim] 추출 완료!

        # ==========================================
        # 3. 원본 믹싱 연산 (states 변수 대신 추출한 state_emb 사용)
        # ==========================================
        agent_qs = agent_qs.view(-1, 1, self.n_agents)
        
        # First layer
        w1 = torch.abs(self.hyper_w_1(state_emb))
        b1 = self.hyper_b_1(state_emb)
        w1 = w1.view(-1, self.n_agents, self.embed_dim)
        b1 = b1.view(-1, 1, self.embed_dim)
        hidden = nn.functional.elu(torch.bmm(agent_qs, w1) + b1)
        
        # Second layer
        w_final = torch.abs(self.hyper_w_final(state_emb))
        w_final = w_final.view(-1, self.embed_dim, 1)
        
        # State-dependent bias
        v = self.V(state_emb).view(-1, 1, 1)
        
        # Compute final output
        y = torch.bmm(hidden, w_final) + v
        
        # Reshape and return
        q_tot = y.view(bs, -1, 1)
        return q_tot