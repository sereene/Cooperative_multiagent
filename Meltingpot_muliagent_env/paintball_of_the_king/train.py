import os
import warnings
import gymnasium as gym
import gc
import torch
import ray
from ray import tune
from ray.tune.registry import register_env
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.models import ModelCatalog
from ray.air.integrations.wandb import WandbLoggerCallback
from datetime import datetime
import tensorflow as tf

# [설정] TF GPU 비활성화 (메모리 절약)
tf.config.set_visible_devices([], 'GPU')
os.environ["TUNE_DISABLE_STRICT_METRIC_CHECKING"] = "1"
warnings.filterwarnings("ignore", category=UserWarning, module="gymnasium.spaces.box")

# 모듈 임포트
from env_utils import env_creator
from model import MeltingPotModel
from callbacks import SelfPlayCallback  # 수정된 콜백 사용

if __name__ == "__main__":
    ray.init(ignore_reinit_error=True)
    ModelCatalog.register_custom_model("meltingpot_model", MeltingPotModel)
    
    env_name = "meltingpot_paintball_koth_mixed"
    register_env(env_name, lambda cfg: env_creator({"substrate": "paintball__king_of_the_hill"}))

    # 환경 스펙 확인
    tmp_env = env_creator({"substrate": "paintball__king_of_the_hill"})
    if hasattr(tmp_env, "possible_agents"):
        agent_list = list(tmp_env.possible_agents)
    else:
        agent_list = list(tmp_env.par_env.possible_agents)
    agent_id = agent_list[0]
    obs_space = tmp_env.par_env.observation_spaces[agent_id]
    act_space = tmp_env.par_env.action_spaces[agent_id]
    tmp_env.close()
    del tmp_env

    # ======================================================================
    # [핵심] Self-Play 정책 정의
    # ======================================================================
    policies = {
        # 1. 메인 정책 (내가 학습할 대상)
        "main_policy": (None, obs_space, act_space, {}),
        
        # 2. 적 정책 (과거의 나 - 학습되지 않음, 주기적으로 복사됨)
        "opponent_policy": (None, obs_space, act_space, {}),
    }

    # 매핑 함수: Red팀은 나(Main), Blue팀은 적(Opponent)
    def policy_mapping_fn(agent_id, *args, **kwargs):
        if agent_id in ["player_0", "player_2"]: # Red Team
            return "main_policy"
        else: # Blue Team (player_1, player_3)
            return "opponent_policy"

    current_dir = os.getcwd()
    local_log_dir = os.path.join(current_dir, "results_selfplay")
    exp_name = "MeltingPot_KOTH_SelfPlay_noBot_1e-5_Fc256"
    start_time = datetime.now().strftime("%m-%d_%H-%M-%S")
    gif_save_path = os.path.join(local_log_dir, exp_name, f"gifs_{start_time}")

    config = (
        PPOConfig()
        .environment(env=env_name, clip_actions=True, disable_env_checking=True)
        .framework("torch")
        .rl_module(_enable_rl_module_api=False)
        .rollouts(
            compress_observations=True,
            # [안전 설정] 워커 4개 (봇 로딩이 없으므로 4개도 충분히 돔)
            num_rollout_workers=8, 
            num_envs_per_worker=1, 
            rollout_fragment_length=256,
            sample_timeout_s=600,
        )
        .training(
            _enable_learner_api=False,
            model={
                "custom_model": "meltingpot_model",
                "max_seq_len": 100,
                "vf_share_layers": False
            },
            lr=1e-5,
            gamma=0.99,
            lambda_=0.95,
            kl_coeff=0.2,
            clip_param=0.2,
            entropy_coeff=0.01, 
            train_batch_size=8*256, 
            sgd_minibatch_size=256,
            num_sgd_iter=10,
        )
        .multi_agent(
            policies=policies,
            policy_mapping_fn=policy_mapping_fn,
            # [중요] 메인 정책만 학습시킴 (메모리 절약 + 안정성)
            policies_to_train=["main_policy"],
        )
        # [Callback] 50 Iteration마다 적을 내 수준으로 업데이트
        .callbacks(lambda: SelfPlayCallback(out_dir=gif_save_path, update_interval_iter=50))
        .evaluation(evaluation_interval=50, evaluation_num_episodes=1, evaluation_config={"explore": False})
        .resources(num_gpus=1 if torch.cuda.is_available() else 0)
    )

    print(f"### Starting Self-Play Training (Main vs Snapshot). Logs: {local_log_dir} ###")

    tune.run(
        "PPO",
        name=exp_name,
        stop={"timesteps_total": 20_000_000},
        local_dir=local_log_dir,
        config=config.to_dict(),
        checkpoint_freq=50, # 체크포인트도 50번마다 저장
        checkpoint_at_end=True,
        keep_checkpoints_num=3,
        checkpoint_score_attr="policy_reward_mean/main_policy",
        metric="policy_reward_mean/main_policy",
        mode="max",
        callbacks=[
            WandbLoggerCallback(
                project="MeltingPot_KOTH_SelfPlay",
                group="Main_vs_Past",
                job_type="training",
                name=exp_name,
                log_config=True
            )
        ]
    )
