import os
import warnings
import torch
import ray
from ray import tune
from ray.tune.registry import register_env
from ray.rllib.algorithms.qmix import QMixConfig  # [변경] QMixConfig 임포트
from ray.air.integrations.wandb import WandbLoggerCallback
from datetime import datetime  

# [변경] grouped_env_creator 임포트
from env_utils import grouped_env_creator 
from callbacks import VideoCallbacks

warnings.filterwarnings("ignore", category=DeprecationWarning)
os.environ["TUNE_DISABLE_STRICT_METRIC_CHECKING"] = "1"

if __name__ == "__main__":
    ray.init()
    
    # [수정] custom_mlp 등록 및 정책 매핑 부분 제거 (QMIX가 내부적으로 알아서 처리함)
    
    env_name = "kaz_QMIX_Vector"
    # [변경] 기존 env_creator 대신 그룹화된 환경 사용
    register_env(env_name, lambda cfg: grouped_env_creator(cfg))
    
    current_dir = os.getcwd()
    local_log_dir = os.path.join(current_dir, "results")
    experiment_name = "KAZ_DQN_VectorObs_2Knights_1e-5"

    start_time = datetime.now().strftime("%m-%d_%H-%M-%S")
    gif_save_path = os.path.join(local_log_dir, experiment_name, f"gifs_{start_time}")

    config = (
        QMixConfig() # [변경] QMIX 설정 사용
        .rl_module(_enable_rl_module_api=False)
        .training(_enable_learner_api=False)
        .environment(env=env_name, clip_actions=True, disable_env_checking=True)
        .framework("torch")
        .rollouts(
            num_rollout_workers=4,
            rollout_fragment_length=4, 
            batch_mode="complete_episodes", # [중요] QMIX는 에피소드 단위 수집이 필수입니다.
            compress_observations=True
        )
        .training(
            mixer="vdn",             # [중요] QMIX 믹서 네트워크 활성화 (또는 "vdn")
            mixing_embed_dim=32,      # 믹서 임베딩 차원
            
            # [중요] QMIX의 train_batch_size 단위는 'Step'이 아니라 '에피소드 개수'입니다.
            train_batch_size=32,      
            target_network_update_freq=500,
            
            lr=1e-5, 
            gamma=0.99,
            
            # # [변경] CustomMLP 대신 RLlib의 기본 모델 사용 (QMIX 구조 호환성을 위해)
            # model={
            #     "fcnet_hiddens": [256, 256, 128],
            #     "fcnet_activation": "relu",
            # }
        )
        .exploration(
            exploration_config={
                "type": "EpsilonGreedy",
                "initial_epsilon": 0.8,
                "final_epsilon": 0.01,       
                "epsilon_timesteps": 10_000_000, 
            }
        )
        .callbacks(lambda: VideoCallbacks(out_dir=gif_save_path, env_creator_fn=grouped_env_creator))
        .evaluation(
            evaluation_interval=100,
            evaluation_num_episodes=25,
            evaluation_config={"explore": False},
        )
        .resources(num_gpus=1 if torch.cuda.is_available() else 0)
    )

    print(f"### Training Logs will be saved at: {local_log_dir} ###")

    tune.run(
        "QMIX", # [변경] DQN -> QMIX
        name=experiment_name,
        stop={"timesteps_total": 20_000_000},
        local_dir=local_log_dir,
        metric="evaluation/custom_metrics/score_mean",
        mode="max",
        keep_checkpoints_num=2,
        checkpoint_score_attr="evaluation/custom_metrics/score_mean",
        checkpoint_freq=200,
        checkpoint_at_end=True,
        config=config.to_dict(),
        callbacks=[
            WandbLoggerCallback(
                project="kaz_multiagent_qmix",
                group="qmix_experiments",
                job_type="training",
                name=experiment_name,
                log_config=True
            )
        ]
    )