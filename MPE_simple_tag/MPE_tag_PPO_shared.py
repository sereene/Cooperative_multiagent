import os
import numpy as np
import gc
import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.tune.registry import register_env
from pettingzoo.mpe import simple_tag_v3
from ray.air.integrations.wandb import WandbLoggerCallback
import supersuit as ss
import torch
import warnings
import gymnasium as gym
from ray.rllib.policy.policy import Policy
import numpy as np
from ray.rllib.models import ModelCatalog
import torch.nn as nn
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from PPO_Independent.models_MLP import FlattenMLP
from RLlib_DQN_Independent import GifCallbacks
import imageio.v2 as imageio


# 경고 메시지 무시
warnings.filterwarnings("ignore", category=DeprecationWarning)
os.environ["TUNE_DISABLE_STRICT_METRIC_CHECKING"] = "1"

# === 파라미터 설정 ===
MAX_CYCLES = 500


class SimpleTagRewardWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
    
    def step(self, actions):
        obs, rewards, terminations, truncations, infos = self.env.step(actions)
        shaped_rewards = rewards.copy()
        
        for agent_id, reward in rewards.items():
            if "adversary" in agent_id:
                if reward > 5: 
                    shaped_rewards[agent_id] += 10.0 # 강한 보상
        
        return obs, shaped_rewards, terminations, truncations, infos


class SimpleTagCallbacks(DefaultCallbacks):
    def on_episode_start(self, *, worker, base_env, policies, episode, **kwargs):
        # 에피소드 시작 시 데이터 초기화
        episode.user_data["prev_rewards"] = {}
        episode.user_data["total_collisions"] = 0

    def on_episode_step(self, *, worker, base_env, episode, **kwargs):
        # 키: (agent_id, policy_id), 값: 누적 보상
        current_rewards = episode.agent_rewards
        prev_rewards = episode.user_data["prev_rewards"]
        
        collisions_this_step = 0
        
        for (agent_id, policy_id), total_reward in current_rewards.items():
            # 추격자(adversary)인 경우만 체크
            if "adversary" in agent_id:
                # 2. 이전 스텝까지의 누적 보상 가져오기 (없으면 0.0)
                prev_total = prev_rewards.get(agent_id, 0.0)
                
                # 3. 이번 스텝에서 받은 보상 = (현재 누적 - 이전 누적)
                step_reward = total_reward - prev_total
                
                # 4. 보상이 5보다 크면 충돌로 간주
                if step_reward > 5:
                    collisions_this_step += 1
                
                # 5. 현재 누적 보상을 저장 (다음 스텝 비교용)
                prev_rewards[agent_id] = total_reward
        
        # 충돌 횟수 누적
        episode.user_data["total_collisions"] += collisions_this_step

    def on_episode_end(self, *, worker, base_env, policies, episode, **kwargs):
        # 에피소드 종료 시 평균 지표로 기록
        total_collisions = episode.user_data.get("total_collisions", 0)
        episode.custom_metrics["collisions"] = total_collisions

# === 2. Callbacks (GIF 저장 로직 수정됨) ===
def rollout_and_save_gif(
    *,
    algorithm,
    out_path: str,
    max_cycles: int,
    every_n_steps: int = 2, # GIF 프레임 수집 간격
    max_frames: int = 300,
    fps: int = 30,
):
    """
    simple_tag 환경에서 에피소드를 실행하며 GIF를 저장합니다.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # 렌더링 모드 켜기
    env = simple_tag_v3.parallel_env(
        num_good=1,
        num_adversaries=2,
        num_obstacles=2,
        max_cycles=max_cycles,
        continuous_actions=False,
        render_mode="rgb_array", # [중요] 픽셀 데이터 반환
    )

    # 관측값 전처리 (학습 환경과 동일하게 맞춰야 함)
    env = ss.pad_observations_v0(env)
    env = ss.pad_action_space_v0(env)

    frames = []
    try:
        obs, infos = env.reset()
        
        # 첫 프레임
        fr0 = env.render()
        if fr0 is not None:
            frames.append(fr0)

        terminations = {a: False for a in env.possible_agents}
        truncations = {a: False for a in env.possible_agents}
        step_i = 0

        while True:
            if all(terminations.values()) or all(truncations.values()):
                break
            
            if len(frames) >= max_frames:
                break

            actions = {}
            for agent_id, agent_obs in obs.items():
                # 정책 매핑 로직 (Main의 policy_mapping_fn과 동일해야 함)
                if "adversary" in agent_id:
                    policy_id = "shared_adversary_policy"
                else:
                    policy_id = "agent_policy" # Heuristic

                action = algorithm.compute_single_action(
                    agent_obs, 
                    policy_id=policy_id,
                    explore=False # 평가 시에는 탐색 끔
                )
                actions[agent_id] = action
                
            obs, rewards, terminations, truncations, infos = env.step(actions)

            # 프레임 수집
            if (step_i % every_n_steps) == 0:
                fr = env.render()
                if fr is not None:
                    frames.append(fr)

            step_i += 1

        if frames:
            imageio.mimsave(out_path, frames, fps=fps)
            print(f"[GIF] saved: {out_path} frames={len(frames)}")
        else:
            print(f"[GIF] skipped (no frames): {out_path}")

    except Exception as e:
        print(f"[Error] generating GIF: {e}")
    finally:
        env.close()
        gc.collect()

class GifCallbacks(SimpleTagCallbacks):
    def __init__(
        self,
        out_dir: str,
        every_n_evals: int = 5,
        max_cycles: int = 100,
        filename_prefix: str = "eval",
    ):
        super().__init__()
        self.out_dir = out_dir
        self.every_n_evals = every_n_evals
        self.max_cycles = max_cycles
        self.filename_prefix = filename_prefix 
        
        self.eval_count = 0
        os.makedirs(self.out_dir, exist_ok=True)

    def on_train_result(self, *, algorithm, result, **kwargs):
        if "evaluation" not in result:
            return

        self.eval_count += 1
        if (self.eval_count % self.every_n_evals) != 0:
            return

        training_iter = int(result.get("training_iteration", 0))
        out_path = os.path.join(
            self.out_dir,
            f"{self.filename_prefix}_{self.eval_count:04d}_iter{training_iter:06d}.gif",
        )

        print(f"Generating GIF at eval step {self.eval_count}...")
        rollout_and_save_gif(
            algorithm=algorithm,
            out_path=out_path,
            max_cycles=self.max_cycles,
        )


# MPE(Vector) 환경 전용 MLP 모델
class FlattenMLP(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        # MPE 환경의 관측값은 1차원 벡터입니다 (예: [x, y, vx, vy, ...]).
        # shape가 (14,) 처럼 들어오므로 shape[0]이 곧 입력 차원입니다.
        input_dim = obs_space.shape[0]
        
        print(f"DEBUG: VectorMLP Init - Input Dim: {input_dim}")

        # 벡터 처리용 MLP 레이어 (Conv2d, MaxPool 제거됨)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),       
            nn.ReLU(),
        )

        self.policy_head = nn.Linear(256, num_outputs)
        self.value_head = nn.Linear(256, 1)
        self._value_out = None

    def forward(self, input_dict, state, seq_lens):
        # 1. 입력 가져오기 (이미지가 아니므로 / 255.0 불필요)
        x = input_dict["obs"].float()

        # [중요] MPE 관측값은 이미 [Batch, Dim] 형태입니다.
        # 차원 확장(unsqueeze)이나 순서 변경(permute)을 하면 안 됩니다.

        # 2. MLP 통과
        x = self.mlp(x)

        # 3. 출력 산출
        logits = self.policy_head(x)
        self._value_out = self.value_head(x).squeeze(-1)
        
        return logits, state

    def value_function(self):
        return self._value_out
    
ModelCatalog.register_custom_model("flatten_mlp", FlattenMLP)


# === 환경 생성 함수 ===
def env_creator(config=None):
    # simple_tag_v3 생성
    # num_good=1 (도망자), num_adversaries=2 (추격자)
    # continuous_actions=False (PPO도 Discrete 액션 지원, MLP 학습에 유리)
    env = simple_tag_v3.parallel_env(
        num_good=1, 
        num_adversaries=2, 
        num_obstacles=2, 
        max_cycles=MAX_CYCLES,
        continuous_actions=False 
    )
    
    # 벡터 관측값 크기 맞추기 (필수)
    env = ss.pad_observations_v0(env)
    env = ss.pad_action_space_v0(env)

    return ParallelPettingZooEnv(env)



# 사용자 정의: 멈추지 않고 무조건 움직이는 정책
class EscapeHeuristicPolicy(Policy):
    def __init__(self, observation_space, action_space, config):
        super().__init__(observation_space, action_space, config)

    def compute_actions(self, obs_batch, state_batches, prev_action_batch=None, prev_reward_batch=None, **kwargs):
        # simple_tag의 Action Space: 0(No-op), 1(Left), 2(Right), 3(Down), 4(Up)
        # 0번(멈춤)을 제외하고 1~4 중에서 랜덤 선택 -> 계속 도망다님
        batch_size = len(obs_batch)
        actions = [np.random.choice([1, 2, 3, 4]) for _ in range(batch_size)]
        
        return np.array(actions), [], {}

    def learn_on_batch(self, samples):
        # 학습하지 않음 (빈 딕셔너리 반환)
        return {}

    def get_weights(self):
        return {}

    def set_weights(self, weights):
        pass


if __name__ == "__main__":
    ray.init()

    env_name = "simple_tag_ppo_shared"
    register_env(env_name, lambda cfg: env_creator(cfg))
    
    # 정책 생성을 위한 임시 환경
    tmp_env = env_creator({})
    obs_space = tmp_env.observation_space
    act_space = tmp_env.action_space
    tmp_env.close()

    # === Shared Policy 설정 ===
    # 1. 추격자(Adversaries) 공유 정책
    # 2. 도망자(Agent) 개별 정책
    policies = {
        "shared_adversary_policy": (None, obs_space, act_space, {}),
        "agent_policy": (EscapeHeuristicPolicy, obs_space, act_space, {}),
    }

    def policy_mapping_fn(agent_id, episode, worker, **kwargs):
        if "adversary" in agent_id:
            return "shared_adversary_policy"
        return "agent_policy"

    # === 로그 저장 경로 ===
    current_dir = os.getcwd()
    local_log_dir = os.path.join(current_dir, "results")
    experiment_name = "PPO_simple_tag_shared_adversary_MLP"
    gif_save_path = os.path.join(local_log_dir, experiment_name, "gifs")


    # === PPO 설정 (기존 파일 설정 유지) ===
    config = (
        PPOConfig()
        .rl_module(_enable_rl_module_api=False)
        .training(_enable_learner_api=False)
        .environment(
            env=env_name, 
            clip_actions=True,
            disable_env_checking=True 
        )
        .framework("torch")
        .rollouts(
            num_rollout_workers=12,        # 병렬 워커 수 유지
            rollout_fragment_length=256,  
            compress_observations=True 
        )
        .multi_agent(
            policies=policies,
            policy_mapping_fn=policy_mapping_fn,
            policies_to_train=["shared_adversary_policy"],
        )
        .training(
            # === 모델 설정: 벡터 입력이므로 MLP 사용 ===
            model={
                "custom_model": "flatten_mlp",
            },
            # === 하이퍼파라미터 (기존 파일 값 유지) ===
            train_batch_size=12 * 256,
            sgd_minibatch_size=256,
            num_sgd_iter=8,
            lr=5e-5,
            gamma=0.99,
            lambda_=0.95,
            use_gae=True,
            vf_loss_coeff=0.5,
            entropy_coeff=0.01,
        )
        # 평가 및 GIF 저장 콜백 연결
        .callbacks(lambda: GifCallbacks(
            out_dir=gif_save_path,
            every_n_evals=3,    # 3번의 evaluation마다 GIF 저장
            max_cycles=MAX_CYCLES,
        ))
        .evaluation(
            evaluation_interval=20,   
            evaluation_num_episodes=20,
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
        stop={"timesteps_total": 10_000_000}, # 5M 스텝 (MPE는 금방 수렴할 수 있음)
        local_dir=local_log_dir,
        config=config.to_dict(),
        metric="evaluation/custom_metrics/collisions_mean",
        mode="max",
        keep_checkpoints_num=2,
        checkpoint_score_attr="evaluation/custom_metrics/collisions_mean",
        checkpoint_freq=50,
        checkpoint_at_end=True,
        
        # WandB 사용 시 (프로젝트 이름만 변경)
        callbacks=[
            WandbLoggerCallback(
                project="simple_tag_multiagent",
                group="ppo_shared_policy",
                job_type="training",
                name=experiment_name,
                log_config=True
            )
        ]
    )