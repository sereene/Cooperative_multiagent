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
from MLPmodels import CustomMLP
from env_utils import env_creator
from callbacks import GifCallbacks

warnings.filterwarnings("ignore", category=DeprecationWarning)
os.environ["TUNE_DISABLE_STRICT_METRIC_CHECKING"] = "1"

if __name__ == "__main__":
    ray.init()
    
    # 커스텀 MLP 모델 등록
    ModelCatalog.register_custom_model("custom_mlp", CustomMLP)

    env_name = "kaz_independent_DoubleDQN_Vector"
    register_env(env_name, lambda cfg: env_creator(cfg))
    
    # [수정] 환경 스펙 확인 부분 변경
    # FixedParallelPettingZooEnv 래퍼 내부의 'par_env'를 통해 접근해야 observation_spaces가 보입니다.
    tmp_env = env_creator({})
    
    # --- 변경된 부분 시작 ---
    obs_space = tmp_env.par_env.observation_spaces["knight_0"] 
    act_space = tmp_env.par_env.action_spaces["knight_0"]
    # --- 변경된 부분 끝 ---
    
    tmp_env.close()

    # [중요] 기사 2명(Independent Learning)을 위한 정책 정의
    policies = {
        "knight_0": (None, obs_space, act_space, {}),
        "knight_1": (None, obs_space, act_space, {}),
    }

    # [중요] Policy Mapping
    def policy_mapping_fn(agent_id, *args, **kwargs): 
        return agent_id

    current_dir = os.getcwd()
    local_log_dir = os.path.join(current_dir, "results")
    experiment_name = "KAZ_Independent_DQN_MLP_VectorObs_2Knights"

    # [수정] 현재 시간을 "년-월-일_시-분-초" 형식으로 가져옴
    start_time = datetime.now().strftime("%m-%d_%H-%M-%S")
    gif_save_path = os.path.join(local_log_dir, experiment_name, f"gifs_{start_time}")

    config = (
        DQNConfig()
        .rl_module(_enable_rl_module_api=False)
        .training(_enable_learner_api=False)
        .environment(env=env_name, clip_actions=True, disable_env_checking=True)
        .framework("torch")
        .rollouts(
            num_rollout_workers=8,
            rollout_fragment_length=64, 
            compress_observations=True
        )
        .training(
            model={"custom_model": "custom_mlp"},
            
            double_q=True, 
            dueling=True, 
            num_atoms=1,
            noisy=False,
            
            replay_buffer_config={
                "type": "MultiAgentReplayBuffer",
                "capacity": 50_000, 
            },
            
            n_step=1,
            target_network_update_freq=10_000,
            train_batch_size=512,
            
            lr=5e-5, 
            gamma=0.99,
        )
        .exploration(
            exploration_config={
                "type": "EpsilonGreedy",
                "initial_epsilon": 0.8,
                "final_epsilon": 0.01,       
                "epsilon_timesteps": 10_000_000, 
            }
        )
        # GIF 콜백 연결
        .callbacks(lambda: GifCallbacks(out_dir=gif_save_path))
        .multi_agent(
            policies=policies,
            policy_mapping_fn=policy_mapping_fn,
            # [중요] 기사 2명 모두 학습 대상에 포함
            policies_to_train=["knight_0", "knight_1"],
        )
        .evaluation(
            evaluation_interval=100,
            evaluation_num_episodes=25,
            evaluation_config={"explore": False},
        )
        .resources(num_gpus=1 if torch.cuda.is_available() else 0)
    )

    print(f"### Training Logs will be saved at: {local_log_dir} ###")

    tune.run(
        "DQN",
        name=experiment_name,
        stop={"timesteps_total": 20_000_000},
        local_dir=local_log_dir,
        
        # [변경] 학습 기준: 점수(Kill)가 가장 높은 모델 저장
        metric="evaluation/custom_metrics/score_mean",
        
        mode="max",
        keep_checkpoints_num=2,
        checkpoint_score_attr="evaluation/custom_metrics/score_mean",
        checkpoint_freq=200,
        checkpoint_at_end=True,
        config=config.to_dict(),
        callbacks=[
            WandbLoggerCallback(
                project="kaz_multiagent_independent",
                group="dqn_experiments",
                job_type="training",
                name=experiment_name,
                log_config=True
            )
        ]
    )