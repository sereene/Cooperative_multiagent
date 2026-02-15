import os
import ray
import torch
import torch.nn as nn
import numpy as np
import wandb
from datetime import datetime
from ray.rllib.algorithms.dqn import DQNConfig
from ray.rllib.models import ModelCatalog
from ray.rllib.policy import Policy
from ray.tune.registry import register_env

# 사용자 모듈 임포트
from MLPmodels import CustomMLP
from FrameStackWrapper import FrameStackWrapper
from env_utils import FixedParallelPettingZooEnv
from callbacks import GifCallbacks
from pettingzoo.butterfly import knights_archers_zombies_v10

# ==============================================================================
# 1. 환경 설정 (기사 1 + 궁수 1)
# ==============================================================================
def mixed_env_creator(config=None):
    env = knights_archers_zombies_v10.parallel_env(
        spawn_rate=50,
        num_archers=1,   # 궁수 1명
        num_knights=1,   # 기사 1명
        max_arrows=10,
        max_cycles=900,
        vector_state=True,
        render_mode="rgb_array"
    )
    env = FrameStackWrapper(env, num_stack=3)
    env = FixedParallelPettingZooEnv(env)
    return env

# ==============================================================================
# 2. [개선됨] 정책 자동 찾기 및 가중치 로드
# ==============================================================================
def find_and_load_weights(target_policy, ckpt_path, agent_keyword):
    """
    체크포인트 경로 내의 policies 폴더를 검색하여,
    agent_keyword('knight' 또는 'archer')가 포함된 정책을 자동으로 찾습니다.
    못 찾으면 첫 번째 폴더(default_policy 등)를 사용합니다.
    """
    policies_dir = os.path.join(ckpt_path, "policies")
    
    if not os.path.exists(policies_dir):
        print(f"[Error] 'policies' folder not found in: {ckpt_path}")
        return

    # 1. 폴더 목록 가져오기
    available_policies = [d for d in os.listdir(policies_dir) if os.path.isdir(os.path.join(policies_dir, d))]
    
    if not available_policies:
        print(f"[Error] No policy folders found in {policies_dir}")
        return

    # 2. 키워드로 매칭되는 정책 찾기
    chosen_id = None
    
    # 2-1. 정확히 키워드가 포함된 폴더 찾기 (예: "knight_0" 안에 "knight" 포함)
    for policy_id in available_policies:
        if agent_keyword in policy_id:
            chosen_id = policy_id
            break
    
    # 2-2. 없으면 첫 번째 폴더 사용 (Fallback)
    if chosen_id is None:
        chosen_id = available_policies[0]
        print(f"[Info] '{agent_keyword}' not found. Falling back to '{chosen_id}'")

    print(f"--> Found policy directory: '{chosen_id}' for agent '{agent_keyword}'")

    # 3. 가중치 로드 로직 (Shape Mismatch 처리)
    full_policy_path = os.path.join(policies_dir, chosen_id)
    temp_policy = Policy.from_checkpoint(full_policy_path)
    source_weights = temp_policy.get_weights()
    del temp_policy 

    target_model = target_policy.model
    current_state_dict = target_model.state_dict()
    new_state_dict = {}

    transferred = 0
    skipped = 0
    
    for key, param in source_weights.items():
        if key in current_state_dict:
            if param.shape == current_state_dict[key].shape:
                new_state_dict[key] = torch.from_numpy(param)
                transferred += 1
            else:
                skipped += 1
    
    target_model.load_state_dict(new_state_dict, strict=False)
    print(f"    [Load Stats] Transferred: {transferred} layers | Skipped (Input/Mismatch): {skipped} layers\n")


# ==============================================================================
# 3. 메인 실행 블록
# ==============================================================================
if __name__ == "__main__":
    # [수정] include_dashboard=False 추가
    ray.init(
        ignore_reinit_error=True, 
        include_dashboard=False  # <--- 대시보드 비활성화
    )
    ModelCatalog.register_custom_model("custom_mlp", CustomMLP)
    
    # ------------------------------------------------------------------
    # [입력] 체크포인트 경로만 정확히 입력하세요. (정책 이름은 자동 검색됨)
    # ------------------------------------------------------------------
    KNIGHT_CKPT_PATH = "/home/jsr/project/Cooperative_pong_RL_agent/KAZ_DQN_Independent/results/KAZ_Independent_DQN_MLP_VectorObs_2Knights/DQN_kaz_independent_DoubleDQN_Vector_5ec4b_00000_0_2026-02-10_03-37-21/checkpoint_000027"
    ARCHER_CKPT_PATH = "/home/jsr/project/Cooperative_pong_RL_agent/KAZ_DQN_Independent/results/KAZ_Independent_DQN_MLP_VectorObs_noStack_knightRewardShaping/DQN_kaz_independent_DoubleDQN_Vector_b3d3e_00000_0_2026-01-20_04-04-38/checkpoint_000032"
    # ------------------------------------------------------------------

    experiment_name = "KAZ_Mixed_Auto_Transfer"
    current_dir = os.getcwd()
    local_log_dir = os.path.join(current_dir, "results_transfer")
    start_time = datetime.now().strftime("%m-%d_%H-%M-%S")
    gif_save_path = os.path.join(local_log_dir, experiment_name, f"gifs_{start_time}")

    # 환경 등록
    env_name = "kaz_mixed_auto_env"
    register_env(env_name, lambda cfg: mixed_env_creator(cfg))

    # 공간 확인
    tmp_env = mixed_env_creator({})
    obs_space_knight = tmp_env.par_env.observation_spaces["knight_0"]
    act_space_knight = tmp_env.par_env.action_spaces["knight_0"]
    obs_space_archer = tmp_env.par_env.observation_spaces["archer_0"]
    act_space_archer = tmp_env.par_env.action_spaces["archer_0"]
    tmp_env.close()

    # 정책 정의
    policies = {
        "policy_knight": (None, obs_space_knight, act_space_knight, {}),
        "policy_archer": (None, obs_space_archer, act_space_archer, {}),
    }

    def policy_mapping_fn(agent_id, *args, **kwargs):
        if "knight" in agent_id: return "policy_knight"
        if "archer" in agent_id: return "policy_archer"
        return "policy_knight"

    # 알고리즘 설정
    print("### Building Algorithm... ###")
    config = (
        DQNConfig()
        .environment(env=env_name, disable_env_checking=True)
        .framework("torch")
        .rollouts(num_rollout_workers=8, rollout_fragment_length=64)
        .training(
            model={"custom_model": "custom_mlp"},
            double_q=True, dueling=True, num_atoms=1, noisy=False,
            replay_buffer_config={"type": "MultiAgentReplayBuffer", "capacity": 50_000},
            n_step=1, target_network_update_freq=10_000,
            train_batch_size=512, lr=5e-5, gamma=0.99,
        )
        .exploration(
            exploration_config={
                "type": "EpsilonGreedy",
                "initial_epsilon": 0.8,
                "final_epsilon": 0.01,
                "epsilon_timesteps": 5_000_000,
            }
        )
        .multi_agent(
            policies=policies,
            policy_mapping_fn=policy_mapping_fn,
            policies_to_train=["policy_knight", "policy_archer"],
        )
        .callbacks(lambda: GifCallbacks(out_dir=gif_save_path, every_n_evals=5))
        .evaluation(
            evaluation_interval=50,
            evaluation_num_episodes=10,
            evaluation_config={"explore": False},
        )
        .resources(num_gpus=1 if torch.cuda.is_available() else 0)
    )

    algo = config.build()

    # ==============================================================================
    # 4. 자동 검색 및 주입 실행
    # ==============================================================================
    print("\n>>> Auto-detecting and injecting weights...")
    
    # 기사 가중치 주입 (키워드 'knight'로 검색)
    find_and_load_weights(
        algo.get_policy("policy_knight"), 
        KNIGHT_CKPT_PATH, 
        agent_keyword="knight"
    )
    
    # 궁수 가중치 주입 (키워드 'archer'로 검색, 없으면 fallback)
    find_and_load_weights(
        algo.get_policy("policy_archer"), 
        ARCHER_CKPT_PATH, 
        agent_keyword="archer"
    )

    # WandB Init
    # [수정] mode="offline" 추가
    wandb.init(
        project="kaz_multiagent_independent",
        group="dqn_transfer_learning",
        name=experiment_name,
        config=config.to_dict(),
        mode="offline"  # <--- 이 줄을 추가하세요!
    )

    print(f"\n### Starting Training Loop (20000 Iterations)... Logs: {local_log_dir} ###")
    
    for i in range(1, 20000):
        result = algo.train()
        
        # 로그 데이터 구성
        mean_reward = result['episode_reward_mean']
        print(f"Iter {i:04d} | Reward: {mean_reward:.2f} | Steps: {result['timesteps_total']}")
        
        log_data = {
            "episode_reward_mean": mean_reward,
            "training_iteration": i,
            "timesteps_total": result['timesteps_total'],
        }
        
        if "policy_reward_mean" in result:
            for pid, rew in result["policy_reward_mean"].items():
                log_data[f"policy_reward_mean/{pid}"] = rew
                
        wandb.log(log_data)

        if i % 100 == 0:
            saved_path = algo.save(os.path.join(local_log_dir, experiment_name))
            print(f"--> Checkpoint saved: {saved_path}")

    wandb.finish()
    algo.stop()
    ray.shutdown()