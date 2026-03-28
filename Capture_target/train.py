import os
import argparse
import ray
from ray import tune
from ray.tune.registry import register_env
from ray.rllib.algorithms.qmix import QMixConfig
from ray.air.integrations.wandb import WandbLoggerCallback
from datetime import datetime

# 작성한 환경 래퍼와 콜백 함수
from env_wrapper import CaptureTargetMultiAgent
from callbacks import VideoCallbacks
import gymnasium as gym

# ==========================================
# 1. 환경 생성 함수 (QMIX용 Grouped 처리)
# ==========================================
def env_creator(env_config):
    env = CaptureTargetMultiAgent(env_config)
    
    # QMIX 학습을 위해 모든 에이전트를 "group_1" 이라는 하나의 Tuple로 묶어줍니다.
    # QMIX 알고리즘의 핵심 조건입니다.
    grouping = {
        "group_1": env.agents
    }
    
    # 그룹의 Observation/Action 공간은 각 에이전트의 공간을 합친 Tuple 형태가 됩니다.
    obs_space = gym.spaces.Tuple([env.observation_space[agent] for agent in env.agents])
    act_space = gym.spaces.Tuple([env.action_space[agent] for agent in env.agents])
    
    # with_agent_groups 메서드를 통해 RLlib 그룹핑 객체로 변환 반환
    grouped_env = env.with_agent_groups(
        grouping, 
        obs_space=obs_space, 
        act_space=act_space
    )
    return grouped_env

# train.py 내부
class CustomVideoCallback(VideoCallbacks):
    def __init__(self):
        super().__init__(
            out_dir=gif_save_path, 
            env_creator_fn=env_creator,
            every_n_evals=5
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_agent", type=int, default=2, help="Number of agents")
    parser.add_argument("--mixer", type=str, default="qmix", choices=["qmix", "vdn"], help="QMIX or VDN")
    args = parser.parse_args()

    ray.init()

    # 환경 등록
    env_name = "CaptureTarget_QMIX_Env"
    register_env(env_name, env_creator)

    current_dir = os.getcwd()
    local_log_dir = os.path.join(current_dir, "results")
    experiment_name = f"QMIX_CaptureTarget_Agent{args.n_agent}"

    start_time = datetime.now().strftime("%m-%d_%H-%M-%S")
    gif_save_path = os.path.join(local_log_dir, experiment_name, f"videos_{start_time}")

    # ==========================================
    # 2. QMIX Config 설정
    # ==========================================
    config = (
        QMixConfig()
        .environment(
            env=env_name,
            env_config={
                "n_agent": args.n_agent,
                "n_target": 1,
                "grid_dim": (8, 8),
                "terminate_step": 60
            },
            disable_env_checking=True
        )
        .framework("torch")
        # Multi-Agent 환경이지만, Grouping 했으므로 Policy는 QMIX가 내부적으로 알아서 생성 및 공유함
        .training(
            mixer=args.mixer,          # "qmix" 또는 "vdn" 선택 가능
            replay_buffer_config={
                "type": "MultiAgentReplayBuffer",  # SimpleReplayBuffer 대신 사용
                "capacity": 50000,                 # 메모리 상황에 따라 조절 (예: 20000~50000)
            },
            mixing_embed_dim=32,       # QMIX 하이퍼파라미터
            train_batch_size=32,       # 환경 스텝 단위 배치 (에피소드 단위가 아님에 주의)
            gamma=0.99,
            lr=1e-4,
        )
        .rollouts(
            num_rollout_workers=8,
            rollout_fragment_length=10, 
        )
        .exploration(
            exploration_config={
                "type": "EpsilonGreedy",
                "initial_epsilon": 1.0,
                "final_epsilon": 0.01,
                "epsilon_timesteps": 1_000_000,
            }
        )
        .evaluation(
            evaluation_interval=10, 
            evaluation_duration=10, 
            evaluation_duration_unit="episodes",
            evaluation_config={
                "explore": False, 
            },
        )
        # 생성한 Video Callback 연동
        .callbacks(CustomVideoCallback)
    )

    print(f"### Training Logs -> {local_log_dir} ###")
    print(f"### Videos -> {gif_save_path} ###")

    # ==========================================
    # 3. Ray Tune 실행
    # ==========================================
    tune.run(
        "QMIX",
        name=experiment_name,
        stop={"timesteps_total": 5_000_000},  # 학습 종료 조건
        local_dir=local_log_dir,
        metric="episode_reward_mean",
        mode="max",
        checkpoint_freq=500,
        checkpoint_at_end=True,
        keep_checkpoints_num=2,
        
        config=config.to_dict(),
        callbacks=[
            WandbLoggerCallback(
                project="CaptureTarget_RL", 
                group="qmix_experiment",
                name=experiment_name,
                log_config=True
            )
        ]
    )