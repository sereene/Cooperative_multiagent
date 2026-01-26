import os
import warnings
import torch
import ray
from ray import tune
from ray.tune.registry import register_env
from ray.rllib.algorithms.dqn import DQNConfig
from ray.rllib.models import ModelCatalog
from ray.air.integrations.wandb import WandbLoggerCallback
from datetime import datetime  

from model import CustomMLP
from env_utils import env_creator
from callbacks import GifCallbacks

warnings.filterwarnings("ignore", category=DeprecationWarning)
os.environ["TUNE_DISABLE_STRICT_METRIC_CHECKING"] = "1"

if __name__ == "__main__":
    # Ray 초기화 (이미 실행 중이면 무시)
    if not ray.is_initialized():
        ray.init()
    
    # 모델 & 환경 등록
    ModelCatalog.register_custom_model("custom_mlp", CustomMLP)
    
    # LBF 환경 ID (설치된 버전에 맞게 수정하세요. 예: -v2 or -v3)
    LBF_ENV_ID = "Foraging-8x8-2p-2f-v3"
    register_env("LBF_env", lambda cfg: env_creator(cfg))
    
    # 환경 스펙 확인
    tmp_env = env_creator({"env_id": LBF_ENV_ID})
    obs_space = tmp_env.observation_space
    act_space = tmp_env.action_space
    tmp_env.close()

    # Shared Policy 설정
    policies = {
        "shared_policy": (None, obs_space, act_space, {}),
    }

    def policy_mapping_fn(agent_id, *args, **kwargs): 
        return "shared_policy"

    # 경로 설정
    current_dir = os.getcwd()
    local_log_dir = os.path.join(current_dir, "results")
    experiment_name = "LBF_Shared_DQN_MLP_noRewardShaping_nofilter"
    start_time = datetime.now().strftime("%m-%d_%H-%M-%S")
    gif_save_path = os.path.join(local_log_dir, experiment_name, f"gifs_{start_time}")

    # --- [Config 설정 시작] ---
    config = (
        DQNConfig()
        .rl_module(_enable_rl_module_api=False)
        .training(_enable_learner_api=False)
        .environment(
            env="LBF_env", 
            env_config={"env_id": LBF_ENV_ID},
            clip_actions=True, 
            disable_env_checking=True,
        )
        .framework("torch")
        .rollouts(
            num_rollout_workers=6,
            rollout_fragment_length=100,
            compress_observations=True
        )
        .training(
            model={"custom_model": "custom_mlp"},
            
            # [핵심 수정 2] n_step=3 적용 (협동 보상 연결을 위해 필수)
            # n_step=3,
            
            # 기본 DQN 설정 유지
            double_q=True, 
            dueling=True, 
            num_atoms=1,
            
            replay_buffer_config={
                "type": "MultiAgentReplayBuffer",
                "capacity": 50_000, 
                # # n_step 사용 시 우선순위 경험 재생(PER) 추천
                # "prioritized_replay_alpha": 0.6,
            },
            
            target_network_update_freq=5000,
            train_batch_size=256,
            lr=5e-5, 
            gamma=0.99,
        )
        .exploration(
            exploration_config={
                "type": "EpsilonGreedy",
                "initial_epsilon": 0.75,
                # [핵심 수정 3] 탐험이 너무 빨리 끝나지 않도록 설정
                # 전체 1000만 스텝 중 500만 스텝까지 탐험 (약 50%)
                "epsilon_timesteps": 10_000_000, 
                # [핵심 수정 4] 최종 탐험 확률 상향 (Local Optima 탈출용)
                "final_epsilon": 0.01,       
            }
        )
        .callbacks(lambda: GifCallbacks(out_dir=gif_save_path, env_name=LBF_ENV_ID))
        .multi_agent(
            policies=policies,
            policy_mapping_fn=policy_mapping_fn,
            policies_to_train=["shared_policy"],
        )
        .evaluation(
            evaluation_interval=50,
            evaluation_num_episodes=20,
            evaluation_config={"explore": False},
        )
        .resources(num_gpus=1 if torch.cuda.is_available() else 0)
    )

    config_dict = config.to_dict()
    config_dict["observation_filter"] = "noFilter" ## 관측치 필터링 비활성화

    print(f"### Training Logs: {local_log_dir} ###")
    print(f"### Experiment: {experiment_name} ###")

    tune.run(
        "DQN",
        name=experiment_name,
        stop={"timesteps_total": 20_000_000},
        local_dir=local_log_dir,
        
        metric="evaluation/custom_metrics/score_mean",
        mode="max",
        
        keep_checkpoints_num=2,
        checkpoint_score_attr="evaluation/custom_metrics/score_mean",
        checkpoint_freq=100,
        checkpoint_at_end=True,
        
        # 수정된 config dict 전달
        config=config_dict, 
        
        callbacks=[
            WandbLoggerCallback(
                project="LBF_shared_policy",
                group="dqn_experiments",
                job_type="training",
                name=experiment_name,
                log_config=True
            )
        ]
    )