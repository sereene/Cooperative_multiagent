import os
import warnings
import torch
import ray
import argparse
import wandb
from ray import tune
from ray.tune.registry import register_env
from ray.rllib.algorithms.dqn import DQNConfig
from ray.rllib.models import ModelCatalog
from ray.air.integrations.wandb import WandbLoggerCallback
from datetime import datetime  
from MLPmodels import CustomMLP
from env_utils import env_creator
from callbacks import GifCallbacks

# [핵심] WandB를 오프라인 모드로 강제 설정
# 인터넷 연결을 시도하지 않고 로컬에만 로그를 쌓습니다.
os.environ["WANDB_MODE"] = "offline"
# 프로세스 충돌 방지
os.environ["WANDB_START_METHOD"] = "thread"

warnings.filterwarnings("ignore", category=DeprecationWarning)
os.environ["TUNE_DISABLE_STRICT_METRIC_CHECKING"] = "1"

class DebugGifCallbacks(GifCallbacks):
    def on_algorithm_init(self, *, algorithm, **kwargs):
        super().on_algorithm_init(algorithm=algorithm, **kwargs)
        print("\n" + "="*50)
        print(f"✅ [WandB 오프라인 모드 시작]")
        print(f"📂 로그 저장소: 로컬 컴퓨터 (나중에 sync 필요)")
        print(f"🚫 인터넷 연결: 사용 안 함 (Crashed 방지)")
        print("="*50 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint")
    parser.add_argument("--wandb_id", type=str, default=None, help="WandB Run ID")
    args = parser.parse_args()

    ray.init()
    
    ModelCatalog.register_custom_model("custom_mlp", CustomMLP)

    env_name = "kaz_independent_DoubleDQN_Vector"
    register_env(env_name, lambda cfg: env_creator(cfg))
    
    tmp_env = env_creator({})
    obs_space = tmp_env.par_env.observation_spaces["knight_0"] 
    act_space = tmp_env.par_env.action_spaces["knight_0"]
    tmp_env.close()

    policies = {
        "knight_0": (None, obs_space, act_space, {}),
        "knight_1": (None, obs_space, act_space, {}),
    }

    def policy_mapping_fn(agent_id, *args, **kwargs): 
        return agent_id

    current_dir = os.getcwd()
    local_log_dir = os.path.join(current_dir, "results")
    experiment_name = "KAZ_Independent_DQN_MLP_VectorObs_2Knights"
    
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
        .callbacks(lambda: DebugGifCallbacks(out_dir=gif_save_path))
        .multi_agent(
            policies=policies,
            policy_mapping_fn=policy_mapping_fn,
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
    
    # 오프라인 설정
    wandb_kwargs = {
        "project": "kaz_multiagent_independent",
        "group": "dqn_experiments",
        "job_type": "training",
        "name": experiment_name,
        "resume": "allow",
        # 오프라인이므로 저장 관련 옵션은 켜도 안전합니다
        "log_config": True,
        "save_checkpoints": False 
    }
    
    if args.wandb_id:
        wandb_kwargs["id"] = args.wandb_id

    tune.run(
        "DQN",
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
        restore=args.checkpoint, 
        callbacks=[
            WandbLoggerCallback(**wandb_kwargs)
        ]
    )