import os
import numpy as np
import ray
import torch
import torch.nn as nn
import supersuit as ss
import imageio.v2 as imageio
import warnings

from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.tune.registry import register_env
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from pettingzoo.butterfly import cooperative_pong_v5

# 사용자가 정의한 Wrapper (파일이 같은 경로에 있어야 함)
from RewardShapingWrapper import RewardShapingWrapper
from MirrorObservationWrapper import MirrorObservationWrapper

# 경고 무시
warnings.filterwarnings("ignore", category=DeprecationWarning)

# ==============================================================================
# 1. Custom Model Definition (학습 코드와 동일해야 함)
# ==============================================================================
class CustomCNN(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        shape = obs_space.shape
        input_channels = shape[2] if len(shape) == 3 else 1
        input_h, input_w = shape[:2]

        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=input_channels, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        dummy_input = torch.zeros(1, input_channels, input_h, input_w)
        with torch.no_grad():
            self.flatten_dim = self.conv_layers(dummy_input).numel()

        self.fc_net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flatten_dim, 512),
            nn.ReLU()
        )

        self.policy_head = nn.Linear(512, num_outputs)
        self.value_head = nn.Linear(512, 1)
        self._value_out = None

    def forward(self, input_dict, state, seq_lens):
        x = input_dict["obs"].float()
        if x.max() > 10.0: x = x / 255.0
        if x.dim() == 3: x = x.unsqueeze(-1)
        x = x.permute(0, 3, 1, 2)

        x = self.fc_net(self.conv_layers(x))
        logits = self.policy_head(x)
        self._value_out = self.value_head(x).squeeze(-1)
        return logits, state

    def value_function(self):
        return self._value_out

# 모델 등록
ModelCatalog.register_custom_model("custom_cnn", CustomCNN)


# ==============================================================================
# 2. Environment Setup (학습 전처리 과정과 동일해야 함)
# ==============================================================================
def make_eval_env(render_mode="rgb_array", max_cycles=500):
    """
    GIF 생성을 위한 환경 생성 함수.
    학습 때 사용한 전처리(Resize 등)를 반드시 포함해야 함.
    """
    env = cooperative_pong_v5.parallel_env(max_cycles=max_cycles, render_mode=render_mode)
    
    # [중요] 학습 코드에 있던 리사이징 (168x84) 적용
    env = ss.resize_v1(env, x_size=168, y_size=84)
    
    # Wrapper 적용
    env = RewardShapingWrapper(env)
    env = MirrorObservationWrapper(env)
    
    return env

def register_rllib_env(env_name="coop_pong_eval"):
    """
    RLlib Algorithm 로딩 시 환경을 찾을 수 있도록 등록하는 함수
    """
    def env_creator(config):
        return ParallelPettingZooEnv(make_eval_env(render_mode=None))
    
    register_env(env_name, env_creator)


# ==============================================================================
# 3. Main Logic
# ==============================================================================
def rollout_checkpoint_to_gif(
    checkpoint_path: str,
    out_gif_path: str,
    num_episodes: int = 1,
    max_cycles: int = 500,
    fps: int = 30,
):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # 1) Ray 초기화
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, include_dashboard=False)

    # 2) 모델 및 환경 등록 (체크포인트 로드 전 필수)
    ModelCatalog.register_custom_model("custom_cnn", CustomCNN)
    register_rllib_env("cooperative_pong_shared_DoubleDQN") # 학습 코드의 환경 이름 사용

    print(f"Loading checkpoint from: {checkpoint_path}")
    
    # 3) 알고리즘 복원
    # 환경 이름이 config에 저장되어 있으므로, 위에서 등록한 이름과 일치해야 함
    algo = Algorithm.from_checkpoint(checkpoint_path)

    # 4) 렌더링용 실제 환경 생성
    env = make_eval_env(render_mode="rgb_array", max_cycles=max_cycles)

    frames = []

    print("Starting rollout...")
    for ep in range(num_episodes):
        obs, infos = env.reset()
        
        # 첫 프레임 렌더링
        fr = env.render()
        if fr is not None: frames.append(fr)

        terminations = {a: False for a in env.possible_agents}
        truncations = {a: False for a in env.possible_agents}
        
        step_i = 0
        while True:
            # 모든 에이전트 종료 체크
            if all(terminations.get(a, False) or truncations.get(a, False) for a in env.possible_agents):
                break
                
            actions = {}
            for agent_id, agent_obs in obs.items():
                # [중요] explore=False로 설정하여 학습된 최적 행동만 선택 (Epsilon=0)
                action = algo.compute_single_action(
                    agent_obs, 
                    policy_id="shared_policy", 
                    explore=False 
                )
                actions[agent_id] = action

            obs, rewards, terminations, truncations, infos = env.step(actions)

            # 프레임 저장
            fr = env.render()
            if fr is not None: frames.append(fr)
            
            step_i += 1
            if step_i >= max_cycles:
                break
        
        print(f"Episode {ep+1} finished with {step_i} steps.")

    env.close()

    # 5) GIF 저장
    if frames:
        os.makedirs(os.path.dirname(out_gif_path) or ".", exist_ok=True)
        imageio.mimsave(out_gif_path, frames, fps=fps)
        print(f"✅ Success! Saved GIF to: {out_gif_path} (Total frames: {len(frames)})")
    else:
        print("❌ Error: No frames captured.")

    algo.stop()
    ray.shutdown()

if __name__ == "__main__":
    # ==========================================================================
    # 👇 경로 설정 부분 (여기를 본인의 체크포인트 경로로 수정하세요)
    # 예: "results/DoubleDQN.../DQN_.../checkpoint_000100" 폴더 경로
    # ==========================================================================
    
    # 예시 경로 (본인의 실제 경로로 변경 필요)
    CHECKPOINT_PATH = "/home/jsr/project/Cooperative_pong_RL_agent/checkpoint_best"
    
    OUTPUT_GIF = "checkpoint_result3.gif"

    # 실행
    # (주의: 경로가 유효하지 않으면 FileNotFoundError 발생)
    try:
        rollout_checkpoint_to_gif(
            checkpoint_path=CHECKPOINT_PATH,
            out_gif_path=OUTPUT_GIF,
            num_episodes=1,
            max_cycles=500,
            fps=30
        )
    except Exception as e:
        print(f"\n[Error Occurred] {e}")
        print("Check if the CHECKPOINT_PATH is correct.")