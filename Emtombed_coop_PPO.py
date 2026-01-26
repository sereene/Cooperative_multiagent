import os
import numpy as np
import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig  # 논문에 따라 PPO 사용
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.tune.registry import register_env
from ray.rllib.models import ModelCatalog
import torch
import torch.nn as nn
from pettingzoo.atari import entombed_cooperative_v3
import supersuit as ss
import imageio.v2 as imageio
from ray.air.integrations.wandb import WandbLoggerCallback

# 경고 메시지 무시
import warnings
from RLlib_DQN_Independent import CustomCNN

warnings.filterwarnings("ignore", category=DeprecationWarning)
os.environ["TUNE_DISABLE_STRICT_METRIC_CHECKING"] = "1"

# 논문에 명시된 총 학습 스텝 수: 10 Million [cite: 250]
TOTAL_TIMESTEPS = 10_000_000
MAX_CYCLES = 2000

# -------------------------------------------------------------------------
# Callbacks (GIF 저장 및 성공 지표)
# -------------------------------------------------------------------------
class EntombedCallbacks(DefaultCallbacks):
    def on_episode_end(self, *, worker, base_env, policies, episode, **kwargs):
        length = episode.length
        # 끝까지 생존하면 성공으로 간주
        success = 1.0 if length >= MAX_CYCLES - 1 else 0.0
        episode.custom_metrics["success"] = success

class GifCallbacks(EntombedCallbacks):
    def __init__(self, out_dir: str, every_n_evals: int = 5, max_cycles: int = 2000):
        super().__init__()
        self.out_dir = out_dir
        self.every_n_evals = every_n_evals
        self.max_cycles = max_cycles
        self.eval_count = 0
        os.makedirs(self.out_dir, exist_ok=True)

    def on_train_result(self, *, algorithm, result, **kwargs):
        training_iter = int(result.get("training_iteration", 0))
        if "evaluation" not in result: return

        self.eval_count += 1
        if (self.eval_count % self.every_n_evals) != 0: return

        out_path = os.path.join(self.out_dir, f"eval_{self.eval_count:04d}_iter{training_iter:06d}.gif")
        self.rollout_and_save_gif(algorithm, out_path)

    def rollout_and_save_gif(self, algorithm, out_path):
        # 환경 생성 (평가용)
        # 전처리 파이프라인은 학습용과 동일해야 함
        env = entombed_cooperative_v3.parallel_env(max_cycles=self.max_cycles, render_mode="rgb_array")
        # env = ss.color_reduction_v0(env, mode='full') # [논문 5.2] Grayscale
        env = ss.resize_v1(env, x_size=96, y_size=96) # [논문 5.2] 96x96 Resize
        # env = ss.frame_stack_v1(env, 4)               # [논문 5.2] Frame Stack 4
        
        # 논문 결과에 따라 Agent Indicator(ID 채널)는 사용하지 않음 ("Identity" method)

        frames = []
        try:
            obs, infos = env.reset()
            frames.append(env.render())
            
            terminations = {a: False for a in env.possible_agents}
            truncations = {a: False for a in env.possible_agents}
            
            step_i = 0
            while not all(terminations.values()) and not all(truncations.values()):
                actions = {}
                for agent_id, agent_obs in obs.items():
                    # Shared Policy 사용
                    action = algorithm.compute_single_action(
                        agent_obs, 
                        policy_id="shared_policy", 
                        explore=False
                    )
                    actions[agent_id] = action
                
                obs, rewards, terminations, truncations, infos = env.step(actions)
                
                if step_i % 4 == 0: # GIF 용량 조절을 위해 4프레임마다 저장
                    frames.append(env.render())
                step_i += 1
                if len(frames) > 500: break # 너무 길어지면 자름

            imageio.mimsave(out_path, frames, fps=30)
            print(f"[GIF] Saved: {out_path}")
        except Exception as e:
            print(f"[GIF] Error: {e}")
        finally:
            env.close()

# -------------------------------------------------------------------------
# Environment Creator
# -------------------------------------------------------------------------
class FixedParallelPettingZooEnv(ParallelPettingZooEnv):
    def __init__(self, env):
        super().__init__(env)
        self.agents = self.par_env.possible_agents
        self._agent_ids = set(self.agents)

    def reset(self, *, seed=None, options=None):
        return self.par_env.reset(seed=seed, options=options)

ModelCatalog.register_custom_model("custom_cnn", CustomCNN)  # 기본 CNN 모델 사용

def env_creator(config=None):
    """
    논문 Section 5.2 Implementation Details에 따른 전처리 적용
    """
    env = entombed_cooperative_v3.parallel_env(max_cycles=MAX_CYCLES, render_mode="rgb_array")
    
    # 1. Grayscale 변환 
    # env = ss.color_reduction_v0(env, mode='full')
    
    # 2. Resize 96x96 
    # 논문에서는 bilinear interpolation을 언급했으나 supersuit 기본값 사용
    env = ss.resize_v1(env, x_size=96, y_size=96)
    
    # 3. Frame Stack (4 frames) 
    # env = ss.frame_stack_v1(env, 4)

    # Agent Indication: "Identity" (아무것도 안 함) 
    # Entombed 환경에서는 에이전트 구분이 없는 것이 성능이 가장 좋았음.

    return FixedParallelPettingZooEnv(env)

# -------------------------------------------------------------------------
# Main Training Script
# -------------------------------------------------------------------------
if __name__ == "__main__":
    ray.init()

    env_name = "entombed_paper_repro"
    register_env(env_name, lambda cfg: env_creator(cfg))
    
    # Observation/Action Space 확인
    tmp_env = env_creator({})
    # PettingZoo Wrapper 특성상 agent 이름을 통해 접근
    agent_0 = tmp_env.agents[0]
    # 함수 호출()이 아니라 속성으로 접근
    obs_space = tmp_env.observation_space 
    act_space = tmp_env.action_space
    tmp_env.close()
    
    print(f"### Observation Space: {obs_space} (Expected: 96x96x4) ###")

    # Shared Policy 정의
    policies = {
        "shared_policy": (None, obs_space, act_space, {}),
    }

    def policy_mapping_fn(agent_id, *args, **kwargs): 
        return "shared_policy"

    current_dir = os.getcwd()
    local_log_dir = os.path.join(current_dir, "results")
    experiment_name = "PPO_Entombed_CNN_Shared_Identity"

    # [논문 Appendix C.1 PPO 하이퍼파라미터 참고]
    # 논문은 Grid Search 범위를 제공했으므로, Atari 표준 PPO 설정을 기반으로 범위 내 값 선택
    config = (
        PPOConfig()
        .rl_module(_enable_rl_module_api=False)
        .training(_enable_learner_api=False)
        .environment(env=env_name, clip_actions=True, disable_env_checking=True)
        .framework("torch")
        .rollouts(
            num_rollout_workers=8, # 병렬 워커 수
            rollout_fragment_length=256, # PPO 표준
        )
        .training(
            model={
                "custom_model": "custom_cnn",
            },
        
            # PPO Hyperparameters 
<<<<<<< HEAD
            lr=2e-5,               # 일반적인 Atari PPO LR
=======
            lr=5e-5,               # 일반적인 Atari PPO LR
>>>>>>> 412078245fd837dfbb4a3cb48757d339382f2fbd
            gamma=0.99,              # Discount Factor [range: 0.9 ~ 0.9999]
            lambda_=0.95,            # GAE Coefficient [range: 0.8 ~ 1.0]
            clip_param=0.1,          # Clip Range [range: 0.1 ~ 0.4]
            entropy_coeff=0.01,      # Entropy Coeff [range: 1e-8 ~ 0.1]
            vf_loss_coeff=0.5,       # Value Function Coeff
            
            train_batch_size=8*256,   # 배치 사이즈 (적절히 조정)
            sgd_minibatch_size=128,  # 미니배치
<<<<<<< HEAD
            num_sgd_iter=8,          # Epochs [range: 1 ~ 20]
            # kwargs={"log_std_init": 0.2}, # 안정적 학습을 위한 초기 로그 표준편차 설정
=======
            num_sgd_iter=4,          # Epochs [range: 1 ~ 20]
>>>>>>> 412078245fd837dfbb4a3cb48757d339382f2fbd
        )
        .callbacks(lambda: GifCallbacks(
            out_dir=os.path.join(local_log_dir, experiment_name, "gifs"),
            max_cycles=MAX_CYCLES
        ))
        .multi_agent(
            policies=policies,
            policy_mapping_fn=policy_mapping_fn,
            policies_to_train=["shared_policy"],
        )
        .evaluation(
            evaluation_interval=50,
<<<<<<< HEAD
            evaluation_num_episodes=20,
=======
            evaluation_num_episodes=5,
>>>>>>> 412078245fd837dfbb4a3cb48757d339382f2fbd
            evaluation_config={"explore": False},
        )
        .resources(num_gpus=1 if torch.cuda.is_available() else 0)
    )

    print(f"### Training Logs will be saved at: {local_log_dir} ###")

    tune.run(
        "PPO",
        name=experiment_name,
        stop={"timesteps_total": TOTAL_TIMESTEPS}, # 10 Million
        local_dir=local_log_dir,
        metric="evaluation/custom_metrics/success_mean",
        mode="max",
        keep_checkpoints_num=2,
        checkpoint_score_attr="evaluation/custom_metrics/success_mean",
        checkpoint_freq=50,
        checkpoint_at_end=True,
        config=config.to_dict(),
        callbacks=[
            WandbLoggerCallback(
                project="entombed_cooperative_shared",
                group="ppo_experiments",
                job_type="training",
                name=experiment_name,
                log_config=True
            )
        ]
    )