import argparse
import os
import warnings
import torch
import ray
from ray import tune
from ray.tune.registry import register_env
from ray.rllib.algorithms.dqn import DQNConfig
from ray.rllib.models import ModelCatalog
from ray.air.integrations.wandb import WandbLoggerCallback
import gymnasium as gym
import numpy as np

# 분리한 모듈들 import
from models import CustomCNN
from env_utils import FixedParallelPettingZooEnv, env_creator
from DQN.callbacks import GifCallbacks

# 경고 무시 및 환경변수 설정
warnings.filterwarnings("ignore", category=DeprecationWarning)
os.environ["TUNE_DISABLE_STRICT_METRIC_CHECKING"] = "1"

if __name__ == "__main__":
    ray.init()
    ModelCatalog.register_custom_model("custom_cnn", CustomCNN)

    # Argument Parser 설정
    parser = argparse.ArgumentParser()
    parser.add_argument("--fc_size", type=int, default=512, help="FC layer hidden size")
    args = parser.parse_args()

    env_name = "cooperative_pong_independent_DoubleDQN"
    register_env(env_name, lambda cfg: env_creator(cfg))
    
    tmp_env = env_creator({})
    obs_space = tmp_env.observation_space
    act_space = tmp_env.action_space
    tmp_env.close()

    # [변경] Independent Policies 설정
    # paddle_0과 paddle_1 각각에 대해 별도의 정책 정의
    policies = {
        "paddle_0": (None, obs_space, act_space, {}),
        "paddle_1": (None, obs_space, act_space, {}),
    }

    # [변경] Policy Mapping 함수: 에이전트 ID가 그대로 정책 이름이 됨
    def policy_mapping_fn(agent_id, *args, **kwargs): 
        return agent_id

    current_dir = os.getcwd()
    local_log_dir = os.path.join(current_dir, "results")
    experiment_name = f"DoubleDQN_CoopPong_Independent_CNN_noRewardShaping_customstack3_fc{args.fc_size}"

    config = (
        DQNConfig()
        .rl_module(_enable_rl_module_api=False)
        .training(_enable_learner_api=False)
        .environment(env=env_name, clip_actions=True, disable_env_checking=True)
        .framework("torch")
        .rollouts(
            num_rollout_workers=10,
            rollout_fragment_length=16, 
            compress_observations=False  #stack 문제 원인 후보
        )
        .training(
            model={
                "custom_model": "custom_cnn",
                
                "custom_model_config": {
                    "fc_size": args.fc_size 
                },
            },

            # --- Double DQN Specifics ---
            double_q=True, 
            dueling=True, 
            num_atoms=1,
            noisy=False,
            
            replay_buffer_config={
                "type": "MultiAgentReplayBuffer",
                "capacity": 50_000, 
            },
            
            n_step=1,
            target_network_update_freq=10000, # 1만 스텝마다 타겟 업데이트 (4)
            train_batch_size=128,
            
            # lr_schedule=[
            #     [0, 5e-5],          # 시작
            #     [3_000_000, 5e-5],  # 300만 스텝까지는 유지 (초반 학습 가속)
            #     [5_000_000, 1e-5],  # 500만 스텝까지 서서히 감소
            #     [10_000_000, 5e-7], # 끝날 때는 아주 낮게
            # ],

            lr=1e-4,  
            gamma=0.95,
        )
        .exploration(
            exploration_config={
                "type": "EpsilonGreedy",
                "initial_epsilon": 1.0,
                "final_epsilon": 0.01,       
                "epsilon_timesteps": 10_000_000, 
            }
        )
        .callbacks(lambda: GifCallbacks(out_dir=os.path.join(local_log_dir, experiment_name, "gifs")))
        .multi_agent(
            policies=policies,
            policy_mapping_fn=policy_mapping_fn,
            # [변경] 두 정책 모두 학습 대상에 포함
            policies_to_train=["paddle_0", "paddle_1"],
        )
        .evaluation(
            evaluation_interval=50,
            evaluation_num_episodes=10, 
            evaluation_config={"explore": False},
        )
        .resources(num_gpus=1 if torch.cuda.is_available() else 0)
    )

    print(f"### Training Logs will be saved at: {local_log_dir} ###")

    tune.run(
        "DQN",
        name=experiment_name,
        stop={"timesteps_total": 25_000_000},
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
                project="cooperative_pong_multiagent_independent", # WandB 프로젝트명 변경 권장
                group="dqn_experiments",
                job_type="training",
                name=experiment_name,
                log_config=True
            )
        ]
    )

    