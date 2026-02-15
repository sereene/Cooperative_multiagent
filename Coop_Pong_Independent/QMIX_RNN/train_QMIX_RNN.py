import argparse
import os
import warnings
import torch
import ray
from ray import tune
from ray.tune.registry import register_env
from ray.rllib.algorithms.qmix import QMixConfig
from ray.rllib.models import ModelCatalog
from ray.air.integrations.wandb import WandbLoggerCallback
import gymnasium as gym
import numpy as np

# 분리한 모듈들 import
from models_CNNGRU import CustomCNNGRU
from env_utils import FixedParallelPettingZooEnv, env_creator
from callbacks_Qmix import GifCallbacks

# 경고 무시 및 환경변수 설정
warnings.filterwarnings("ignore", category=DeprecationWarning)
os.environ["TUNE_DISABLE_STRICT_METRIC_CHECKING"] = "1"

if __name__ == "__main__":
    ray.init()
    # CustomCNNGRU 등록
    ModelCatalog.register_custom_model("custom_cnn_gru", CustomCNNGRU)

    parser = argparse.ArgumentParser()
    # 논문 하이퍼파라미터: Hidden Dimension = 64
    parser.add_argument("--fc_size", type=int, default=64, help="Hidden dimension size (paper: 64)")
    args = parser.parse_args()

    # [중요] QMIX 그룹 정의
    grouping = {
        "group_1": ["paddle_0", "paddle_1"],
    }

    # 환경 생성 함수
    def grouped_env_creator(config):
        env = env_creator(config)
        # env.observation_space는 FixedParallelPettingZooEnv에서 Box 객체로 정의되어 있으므로 그대로 전달
        return env.with_agent_groups(
            grouping,
            obs_space=env.observation_space, 
            act_space=env.action_space
        )

    env_name = "cooperative_pong_qmix"
    register_env(env_name, grouped_env_creator)
    
    # --------------------------------------------------------------------------
    # [수정됨] 그룹화된 공간(Tuple)을 수동으로 생성합니다.
    # --------------------------------------------------------------------------
    # 1. 기본 환경을 생성하여 단일 에이전트의 공간을 가져옵니다.
    base_env = env_creator({})
    single_obs_space = base_env.observation_space
    single_act_space = base_env.action_space
    base_env.close()

    # 2. 그룹은 2명의 에이전트("paddle_0", "paddle_1")로 구성되므로
    #    관측/행동 공간도 2개짜리 Tuple로 만들어줍니다.
    group_obs_space = gym.spaces.Tuple([single_obs_space, single_obs_space])
    group_act_space = gym.spaces.Tuple([single_act_space, single_act_space])
    # --------------------------------------------------------------------------

    current_dir = os.getcwd()
    local_log_dir = os.path.join(current_dir, "results")
    experiment_name = f"QMIX_CoopPong_RNN_hidden{args.fc_size}_5e-5"

    config = (
        QMixConfig()
        .rl_module(_enable_rl_module_api=False)
        .training(_enable_learner_api=False)
        .environment(
            env=env_name,
            disable_env_checking=True # Wrapper 구조 오인식 방지
        )
        .framework("torch")
        .rollouts(
            num_rollout_workers=8, 
            rollout_fragment_length=4, 
        )
        # [핵심] Multi-Agent 정책에 위에서 만든 Tuple 공간을 명시합니다.
        .multi_agent(
            policies={
                "group_1": (None, group_obs_space, group_act_space, {})
            },
            policy_mapping_fn=lambda agent_id, *args, **kwargs: "group_1"
        )
        .training(
            # --- QMIX Specifics ---
            mixer="qmix", 
            mixing_embed_dim=32,
            double_q=True, 
            
            # 모델 설정: CustomCNNGRU 사용
            model={
                "custom_model": "custom_cnn_gru",
                "custom_model_config": {
                    "fc_size": args.fc_size 
                },
                "max_seq_len": 900, 
            },
            
            train_batch_size=32, 
            target_network_update_freq=200,
            replay_buffer_config={
                "type": "MultiAgentReplayBuffer",
                "capacity": 5000,
            },
            
            lr=2e-4,  
            gamma=0.99,
        )
        .training(
        mixer="qmix", 
        mixing_embed_dim=32,
        double_q=True, 
        grad_clip=20.0, 
        
        lr=5e-5, 
        
        target_network_update_freq=1000, 
        
        train_batch_size=64, 
        
        model={
            "custom_model": "custom_cnn_gru",
            "custom_model_config": {"fc_size": args.fc_size},
            "max_seq_len": 900,
        },
    )
        .evaluation(
            evaluation_interval=10,          # 10 Iteration마다 평가 실행
            evaluation_duration=5,          # 평가 시 5 에피소드 진행
            evaluation_duration_unit="episodes",
            evaluation_config={
                "explore": False,           # 평가 시에는 Greedy하게 행동
            },
        )
        .exploration(
            exploration_config={
                "type": "EpsilonGreedy",
                "initial_epsilon": 1.0,
                "final_epsilon": 0.05,       
                "epsilon_timesteps": 10_000_000, 
            }
        )
        .callbacks(lambda: GifCallbacks(out_dir=os.path.join(local_log_dir, experiment_name, "gifs")))
        .resources(num_gpus=1 if torch.cuda.is_available() else 0)
    )

    print(f"### Training Logs will be saved at: {local_log_dir} ###")

    tune.run(
        "QMIX",
        name=experiment_name,
        stop={"timesteps_total": 20_000_000},
        local_dir=local_log_dir,
        metric="evaluation/custom_metrics/success_mean",
        mode="max",
        checkpoint_freq=500,
        checkpoint_at_end=True,
        config=config.to_dict(),
        callbacks=[
            WandbLoggerCallback(
                project="cooperative_pong_algorithm_comparison", 
                group="qmix_paper_repro",
                job_type="training",
                name=experiment_name,
                log_config=True
            )
        ]
    )