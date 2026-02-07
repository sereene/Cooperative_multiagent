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

# [설정] TF GPU 비활성화
tf.config.set_visible_devices([], 'GPU')
os.environ["TUNE_DISABLE_STRICT_METRIC_CHECKING"] = "1"
warnings.filterwarnings("ignore", category=UserWarning, module="gymnasium.spaces.box")

from env_utils import env_creator
from CNN_LSTM_model import MeltingPotModel
from callbacks import SelfPlayCallback

if __name__ == "__main__":
    ray.init(ignore_reinit_error=True)
    ModelCatalog.register_custom_model("meltingpot_model", MeltingPotModel)
    
    env_name = "meltingpot_paintball_koth_mixed"
    register_env(env_name, lambda cfg: env_creator({"substrate": "paintball__king_of_the_hill"}))

    # WandB 설정 정보
    WANDB_PROJECT = "MeltingPot_KOTH_SelfPlay"
    WANDB_GROUP = "Main_vs_Past"
    EXP_NAME = "MeltingPot_KOTH_SelfPlay_noBot_1e-5_lstm_fc128"

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

    # Self-Play 정책 정의
    policies = {
        "main_policy": (None, obs_space, act_space, {}),
        "opponent_policy": (None, obs_space, act_space, {}),
    }

    def policy_mapping_fn(agent_id, *args, **kwargs):
        if agent_id in ["player_0", "player_2"]: # Red Team
            return "main_policy"
        else: # Blue Team (player_1, player_3)
            return "opponent_policy"

    current_dir = os.getcwd()
    local_log_dir = os.path.join(current_dir, "results_selfplay")
    
    # [복구] 타임스탬프가 포함된 GIF 저장 경로 생성
    start_time = datetime.now().strftime("%m-%d_%H-%M-%S")
    gif_save_path = os.path.join(local_log_dir, EXP_NAME, f"gifs_{start_time}")
    os.makedirs(gif_save_path, exist_ok=True) # 경로 미리 생성

    config = (
        PPOConfig()
        .environment(env=env_name, clip_actions=True, disable_env_checking=True)
        .framework("torch")
        .rl_module(_enable_rl_module_api=False)
        .rollouts(
            compress_observations=True,
            num_rollout_workers=8, 
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
            policies_to_train=["main_policy"],
        )
        # [복구] GIF 저장 경로를 콜백에 전달
        .callbacks(lambda: SelfPlayCallback(
            out_dir=gif_save_path,
            update_interval_iter=20,
            max_cycles=1000
        ))
        .evaluation(evaluation_interval=50, evaluation_num_episodes=1, evaluation_config={"explore": False})
        .resources(num_gpus=1 if torch.cuda.is_available() else 0)
    )

    print(f"### Starting Self-Play Training. Logs: {local_log_dir} ###")
    print(f"### Local GIFs will be saved to: {gif_save_path} ###")

    tune.run(
        "PPO",
        name=EXP_NAME,
        stop={"timesteps_total": 20_000_000},
        local_dir=local_log_dir,
        config=config.to_dict(),
        checkpoint_freq=50, 
        checkpoint_at_end=True,
        keep_checkpoints_num=3,
        checkpoint_score_attr="training_iteration",
        metric="training_iteration",
        mode="max",
        # 표준 WandB 로거 사용 (메트릭 로깅용)
        callbacks=[
            WandbLoggerCallback(
                project=WANDB_PROJECT,
                group=WANDB_GROUP,
                name=EXP_NAME,
                log_config=True
            )
        ]
    )
