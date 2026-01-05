import gc
import os
import numpy as np
import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.tune.registry import register_env
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
import torch
import torch.nn as nn
from pettingzoo.butterfly import cooperative_pong_v5
from ray.air.integrations.wandb import WandbLoggerCallback
import gymnasium as gym
import warnings
import supersuit as ss  
from MirrorObservationWrapper import MirrorObservationWrapper
from RewardShapingWrapper import RewardShapingWrapper
import os
import imageio.v2 as imageio



# 경고 메시지 무시
warnings.filterwarnings("ignore", category=DeprecationWarning)

# MAX_CYCLES = 900


# 평가 지표 누락 시 에러 무시
os.environ["TUNE_DISABLE_STRICT_METRIC_CHECKING"] = "1"

MAX_CYCLES = 500

class CoopPongCallbacks(DefaultCallbacks):
    def on_episode_end(self, *, worker, base_env, policies, episode, **kwargs):
        length = episode.length
        success = 1.0 if length >= MAX_CYCLES - 1 else 0.0
        episode.custom_metrics["success"] = success


def rollout_and_save_gif(
    *,
    algorithm,
    out_path: str,
    max_cycles: int,
    every_n_steps: int = 4,
    max_frames: int = 200,
    fps: int = 30,
):
    """
    algorithm(현재 학습 중인 PPO/Algo 인스턴스)을 이용해
    cooperative_pong_v5에서 1 에피소드 롤아웃하며 rgb_array 렌더 프레임을 모아 GIF 저장.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    env = cooperative_pong_v5.parallel_env(
        max_cycles=max_cycles,
        render_mode="rgb_array",
    )

    # # 1. 크기 줄이기 (AI에게는 84x84로 보임, 하지만 render() 결과는 고화질일 수 있음)
    # env = ss.resize_v1(env, x_size=168, y_size=84)
    
    # 2. 흑백 변환
    # env = ss.color_reduction_v0(env, mode='full')
    
    # # 3. 프레임 스택 (4장 겹치기)
    # env = ss.frame_stack_v1(env, 4)
    
    env = RewardShapingWrapper(env)

    # env = MirrorObservationWrapper(env)

    frames = []
    try:
        obs, infos = env.reset()
        step_i = 0

        # 첫 프레임
        fr0 = env.render()
        if fr0 is not None:
            frames.append(fr0)

        terminations = {a: False for a in env.possible_agents}
        truncations = {a: False for a in env.possible_agents}

        while True:
            if all(
                terminations.get(a, False) or truncations.get(a, False)
                for a in env.possible_agents
            ):
                break

            actions = {}
            for agent_id, agent_obs in obs.items():

                action = algorithm.compute_single_action(agent_obs, policy_id="shared_policy")
                actions[agent_id] = action

            obs, rewards, terminations, truncations, infos = env.step(actions)

            if (step_i % every_n_steps) == 0:
                if len(frames) >= max_frames:
                    break
                fr = env.render()
                if fr is not None:
                    frames.append(fr)

            step_i += 1

        if frames:
            imageio.mimsave(out_path, frames, fps=fps)
            print(f"[GIF] saved: {out_path} frames={len(frames)}")
        else:
            print(f"[GIF] skipped (no frames): {out_path}")

    finally:
        try:
            env.close()
            gc.collect()
        except Exception:
            pass


class GifCallbacks(CoopPongCallbacks):
    """
    evaluation 5번마다 GIF 1개 저장 

    사용 예)
      .callbacks(lambda: EvalEveryNGifCallbacks(
          out_dir=".../results/gifs",
          every_n_evals=5,
          max_cycles=500,
      ))
    """

    def __init__(
        self,
        out_dir: str,
        every_n_evals: int = 5,
        max_cycles: int = 500,
        every_n_steps: int = 2,
        max_frames: int = 400,
        fps: int = 30,
        filename_prefix: str = "eval5",
    ):
        super().__init__()
        self.out_dir = out_dir
        self.every_n_evals = every_n_evals
        self.max_cycles = max_cycles
        self.every_n_steps = every_n_steps
        self.max_frames = max_frames
        self.fps = fps
        self.filename_prefix = filename_prefix

        self.eval_count = 0
        os.makedirs(self.out_dir, exist_ok=True)

    def on_train_result(self, *, algorithm, result, **kwargs):
        """
        RLlib에서 evaluation이 돌면 보통 result 안에 "evaluation" 키가 포함됨.
        (evaluation_interval 설정되어 있을 때)
        """
        training_iter = int(result.get("training_iteration", 0))

        if "evaluation" not in result:
            return

        self.eval_count += 1
        if (self.eval_count % self.every_n_evals) != 0:
            return

        out_path = os.path.join(
            self.out_dir,
            f"{self.filename_prefix}_{self.eval_count:04d}_iter{training_iter:06d}.gif",
        )

        rollout_and_save_gif(
            algorithm=algorithm,
            out_path=out_path,
            max_cycles=self.max_cycles,
            every_n_steps=self.every_n_steps,
            max_frames=self.max_frames,
            fps=self.fps,
        )


class FixedParallelPettingZooEnv(ParallelPettingZooEnv):
    def __init__(self, env):
        super().__init__(env)
        self.agents = self.par_env.possible_agents
        self._agent_ids = set(self.agents)

    def reset(self, *, seed=None, options=None):
        obs, info = self.par_env.reset(seed=seed, options=options)
        return obs, info

def env_creator(config=None):

    env = cooperative_pong_v5.parallel_env(
        max_cycles=MAX_CYCLES,
        render_mode="rgb_array"
    )
    env = ss.max_observation_v0(env,2)  # 최근 2프레임 중 픽셀값 최대치 취함

    # # 가로 168, 세로 84 크기로 줄이기
    # env = ss.resize_v1(env, x_size=168, y_size=84)

    # # 흑백으로 변환하면 메모리를 1/3 더 줄일 수 있음
    # env = ss.color_reduction_v0(env, mode="full")

    # # SuperSuit으로 프레임 4개 강제 스택 (채널이 3 -> 12로 바뀜)
    # env = ss.frame_stack_v1(env, 4)

    # reward shaping wrapper 추가
    env = RewardShapingWrapper(env)

    # Mirror Observation Wrapper 추가
    # env = MirrorObservationWrapper(env)

    return FixedParallelPettingZooEnv(env)


# class CustomCNN(TorchModelV2, nn.Module):
#     def __init__(self, obs_space, action_space, num_outputs, model_config, name):
#         TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
#         nn.Module.__init__(self)

#         # 입력 이미지 차원 자동 감지 (흑백/컬러 호환)
#         # obs_space.shape가 (H, W, C)일 수도, 흑백의 경우 (H, W)
#         shape = obs_space.shape
#         if len(shape) == 3:
#             input_h, input_w, input_channels = shape
#         elif len(shape) == 2:
#             # (H, W)인 경우 채널은 1로 간주
#             input_h, input_w = shape
#             input_channels = 1
#         else:
#             raise ValueError(f"Unsupported observation shape: {shape}")

#         print(f"DEBUG: CNN Init - Input Shape: ({input_h}, {input_w}, {input_channels})")

#         # CNN 레이어 정의 (기존과 동일하지만 input_channels 변수가 1로 들어감)
#         self.conv_layers = nn.Sequential(
#             # Layer 1: 큰 특징 잡기 (8x8 커널)
#             # 입력: 84x84 -> 출력: 20x20
#             nn.Conv2d(in_channels=input_channels, out_channels=32, kernel_size=8, stride=4),
#             nn.ReLU(),

#             # Layer 2: 중간 특징 잡기 (4x4 커널)
#             # 입력: 20x20 -> 출력: 9x9
#             nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
#             nn.ReLU(),
            
#             # Layer 3: 세밀한 특징 잡기 (3x3 커널)
#             # 입력: 9x9 -> 출력: 7x7
#             nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
#             nn.ReLU()
#         )

#         # [수정 2] Dummy Forward로 Flatten 차원 계산
#         # 흑백 이미지(채널1)에 맞는 dummy 텐서 생성
#         dummy_input = torch.zeros(1, input_channels, input_h, input_w)
#         with torch.no_grad():
#             dummy_out = self.conv_layers(dummy_input)
#             self.flatten_dim = dummy_out.numel() 
        
#         print(f"DEBUG: CNN Flatten Dim calculated: {self.flatten_dim}")

#         # FC 레이어
#         self.fc_net = nn.Sequential(
#             nn.Flatten(),
#             nn.Linear(self.flatten_dim, 512),
#             nn.ReLU()
#         )

#         self.policy_head = nn.Linear(512, num_outputs)
#         self.value_head = nn.Linear(512, 1)
        
#         self._value_out = None

#     def forward(self, input_dict, state, seq_lens):
        
#         x = input_dict["obs"].float()

#         if x.max() > 10.0: # 안전장치
#             x = x / 255.0 # 1. 입력 전처리: 정규화
        
#         # 입력 x의 shape:
#         #   - 컬러 혹은 채널 유지된 흑백: [Batch, H, W, C]
#         #   - 채널이 압축된 흑백: [Batch, H, W]
        
#         # 만약 [Batch, H, W]라면 [Batch, H, W, 1]로 늘려줌
#         if x.dim() == 3: 
#             x = x.unsqueeze(-1)
        
#         # [Batch, H, W, C] -> [Batch, C, H, W] 로 변경 (PyTorch Conv2d 순서)
#         x = x.permute(0, 3, 1, 2)

#         # 2. CNN 통과
#         x = self.conv_layers(x)
        
#         # 3. FC 통과
#         x = self.fc_net(x)

#         # 4. 출력 산출
#         logits = self.policy_head(x)
#         self._value_out = self.value_head(x).squeeze(-1)
        
#         return logits, state

#     def value_function(self):
#         return self._value_out
    

# # 모델 등록 이름 변경 (custom_cnn)
# ModelCatalog.register_custom_model("custom_cnn", CustomCNN)

# Flatten MLP 모델
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
    

ModelCatalog.register_custom_model("flatten_mlp", FlattenMLP)

if __name__ == "__main__":
    ray.init()

    env_name = "cooperative_pong_shared_reward_shaping_MLP"
    register_env(env_name, lambda cfg: env_creator(cfg))
    
    tmp_env = env_creator({})
    obs_space = tmp_env.observation_space
    act_space = tmp_env.action_space
    tmp_env.close()

    # Shared Policy 정의: 하나의 정책("shared_policy")만 생성
    policies = {
        "shared_policy": (None, obs_space, act_space, {}),
    }

    # Policy Mapping 함수: 모든 에이전트 ID를 "shared_policy"로 매핑
    def policy_mapping_fn(agent_id, *args, **kwargs):
        return "shared_policy"


    # 각 에이전트 별로 별도 정책 사용
    # policies = {
    #     "policy_left": (None, obs_space, act_space, {}),
    #     "policy_right": (None, obs_space, act_space, {}),
    # }

    # def policy_mapping_fn(agent_id, *args, **kwargs):
    #     return "policy_left" if "0" in agent_id else "policy_right"

    current_dir = os.getcwd()
    local_log_dir = os.path.join(current_dir, "results")

    experiment_name = "PPO_cooperative_pong_newRewardShaping_shared_512_5e-5_MLP_non_reduction"

    gif_save_path = os.path.join(local_log_dir, experiment_name, "gifs")

    config = (
        PPOConfig()
        .rl_module(_enable_rl_module_api=False)
        .training(
            _enable_learner_api=False
        )
        .environment(
            env=env_name, 
            clip_actions=True,
            disable_env_checking=True 
        )
        .framework("torch")
        .rollouts(
            num_rollout_workers=6,
            rollout_fragment_length=256,
            compress_observations=True 
        )
        .training(
            model={
                "custom_model": "flatten_mlp",
            },
            train_batch_size=6 * 256,
            sgd_minibatch_size=256,
            num_sgd_iter=8,
            lr=5e-5, # 2e-6
            gamma=0.99,
            lambda_=0.95,
            use_gae=True,
            vf_loss_coeff=0.5,
            entropy_coeff=0.01,
        )
        .callbacks(lambda: GifCallbacks(
          out_dir=gif_save_path,
          every_n_evals=5,
          max_cycles=500,
        ))
        # .callbacks(CoopPongCallbacks)
        .multi_agent(
            policies=policies,
            policy_mapping_fn=policy_mapping_fn,
        )
        .evaluation(
            evaluation_interval=100,
            evaluation_num_episodes=25,
            evaluation_config={
                "explore": False,
            },
        )
        .resources(
            num_gpus=1 if torch.cuda.is_available() else 0
        )
    )

    print(f"### Training Logs will be saved at: {local_log_dir} ###")

    tune.run(
        "PPO",
        name=experiment_name,
        stop={"timesteps_total": 20_000_000},
        local_dir=local_log_dir,
        metric="evaluation/custom_metrics/success_mean",
        mode="max",
        keep_checkpoints_num=2,
        checkpoint_score_attr="evaluation/custom_metrics/success_mean",
        checkpoint_freq=100,
        checkpoint_at_end=True,
        config=config.to_dict(),

        callbacks=[
            WandbLoggerCallback(
                project="cooperative_pong_multiagent_shared_policy",
                group="ppo_experiments",
                job_type="training",
                name=experiment_name,
                log_config=True
            )
        ]
    )