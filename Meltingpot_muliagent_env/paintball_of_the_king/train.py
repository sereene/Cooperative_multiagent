import os
# 메트릭 누락 에러 방지 설정
os.environ["TUNE_DISABLE_STRICT_METRIC_CHECKING"] = "1"
import warnings
import gymnasium as gym
import gc

# "Casting input x to numpy array" 경고 무시하기
warnings.filterwarnings("ignore", category=UserWarning, module="gymnasium.spaces.box")

# [변경] 봇 정책(TFSavedModelPolicy)은 이제 필요 없으므로 사용하지 않지만,
# 혹시 나중에 다시 쓸 수도 있으니 import는 두되 사용은 안 합니다.
from savedModelPolicy import TFSavedModelPolicy 

import torch
import ray
from ray import tune
from ray.tune.registry import register_env
from ray.rllib.algorithms.ppo import PPOConfig 
from ray.rllib.models import ModelCatalog
from ray.rllib.policy.policy import Policy
from ray.air.integrations.wandb import WandbLoggerCallback
from datetime import datetime
import tensorflow as tf

# TF는 CPU만 쓰도록 강제 (메모리 절약)
tf.config.set_visible_devices([], 'GPU')

# 모듈 임포트
from env_utils import env_creator
from model import MeltingPotModel
from callbacks import GifCallbacks

# [메모리 누수 방지용 콜백]
class MemoryCleanupCallback(GifCallbacks):
    def on_train_result(self, *, algorithm, result, **kwargs):
        # 기존 GIF 생성 로직 수행
        super().on_train_result(algorithm=algorithm, result=result, **kwargs)
        
        # [핵심] 학습 한 번 끝날 때마다 강제로 메모리 청소
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == "__main__":
    ray.init(ignore_reinit_error=True)
    ModelCatalog.register_custom_model("meltingpot_model", MeltingPotModel)
    
    env_name = "meltingpot_paintball_koth_mixed"
    register_env(env_name, lambda cfg: env_creator({"substrate": "paintball__king_of_the_hill"}))

    # [스펙 확인]
    print("DEBUG: Checking Environment Specs...")
    tmp_env = env_creator({"substrate": "paintball__king_of_the_hill"})
    if hasattr(tmp_env, "possible_agents"):
        agent_list = list(tmp_env.possible_agents)
    else:
        agent_list = list(tmp_env.par_env.possible_agents)
    agent_id = agent_list[0]

    obs_space = tmp_env.par_env.observation_spaces[agent_id]
    act_space = tmp_env.par_env.action_spaces[agent_id]
    tmp_env.close()

    # ======================================================================
    # [핵심 변경 1] 정책 정의: 봇을 없애고 Red/Blue 정책 2개를 만듭니다.
    # ======================================================================
    policies = {
        # Team Red: Player 0, 2
        "red_policy": (None, obs_space, act_space, {}),
        
        # Team Blue: Player 1, 3
        "blue_policy": (None, obs_space, act_space, {}),
    }

    # ======================================================================
    # [핵심 변경 2] 매핑 함수: 팀별로 정책을 배정합니다.
    # ======================================================================
    def policy_mapping_fn(agent_id, *args, **kwargs):
        if agent_id in ["player_0", "player_2"]:
            return "red_policy"
        elif agent_id in ["player_1", "player_3"]:
            return "blue_policy"
        else:
            # 혹시 모를 예외 처리 (기본적으로 Red로 보냄)
            return "red_policy"

    current_dir = os.getcwd()
    local_log_dir = os.path.join(current_dir, "results_ppo_mixed")
    # 실험 이름 변경 (SelfPlay 명시)
    exp_name = "MeltingPot_KOTH_PPO_SelfPlay_2vs2_Light" 
    start_time = datetime.now().strftime("%m-%d_%H-%M-%S")
    gif_save_path = os.path.join(local_log_dir, exp_name, f"gifs_{start_time}")

    # [PPO 설정]
    config = (
        PPOConfig()
        .environment(env=env_name, clip_actions=True, disable_env_checking=True)
        .framework("torch")
        .rl_module(_enable_rl_module_api=False)
        .rollouts(
            compress_observations=True,

            num_rollout_workers=8,
            rollout_fragment_length=256
        )
        .training(
            _enable_learner_api=False,
            model={
                "custom_model": "meltingpot_model",
                "max_seq_len": 100,
                "vf_share_layers": False
            },
            lr=5e-6,
            gamma=0.99,
            lambda_=0.95,
            kl_coeff=0.2,
            clip_param=0.2,
            entropy_coeff=0.01, 
            train_batch_size=2048, # 256 * 8
            sgd_minibatch_size=256,
            num_sgd_iter=10,
        )
        .multi_agent(
            policies=policies,
            policy_mapping_fn=policy_mapping_fn,
            # [핵심 변경 3] 두 정책 모두 학습 대상에 포함시킵니다.
            policies_to_train=["red_policy", "blue_policy"],
        )
        .callbacks(lambda: MemoryCleanupCallback(out_dir=gif_save_path, max_cycles=1000))
        .evaluation(evaluation_interval=50, evaluation_num_episodes=1, evaluation_config={"explore": False})
        .resources(num_gpus=1 if torch.cuda.is_available() else 0)
    )

    print(f"### Starting PPO Self-Play Training. Logs: {local_log_dir} ###")

    tune.run(
        "PPO",
        name=exp_name,
        stop={"timesteps_total": 20_000_000},
        local_dir=local_log_dir,
        config=config.to_dict(),
          
        checkpoint_freq=100,
        checkpoint_at_end=True,
        keep_checkpoints_num=2,
        # 점령률(Red팀 기준)을 메트릭으로 사용
        checkpoint_score_attr="custom_metrics/total_zaps_in_episode",
        metric="custom_metrics/total_zaps_in_episode",
        mode="max",

        callbacks=[
            WandbLoggerCallback(
                project="MeltingPot_KOTH",
                group="PPO_SelfPlay",
                job_type="training",
                name=exp_name,
                log_config=True
            )
        ]
    )