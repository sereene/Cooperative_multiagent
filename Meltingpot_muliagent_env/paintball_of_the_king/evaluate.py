import os
import cv2  # 화면 렌더링용
import numpy as np
import ray
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env

# 사용자가 업로드한 파일에서 모듈 임포트
from env_utils import env_creator
from model import MeltingPotModel

# ==============================================================================
# [설정] 체크포인트 경로 및 옵션
# ==============================================================================
# 예: "results_selfplay/MeltingPot_KOTH_.../checkpoint_000050"
# 폴더 안에 'algorithm_state.pkl' 또는 'rllib_checkpoint.json' 등이 들어있는 경로여야 합니다.
CHECKPOINT_PATH = "/home/jsr/project/Cooperative_pong_RL_agent/Meltingpot_muliagent_env/paintball_of_the_king/results_selfplay/MeltingPot_KOTH_SelfPlay_noBot_1e-5_Fc256/PPO_meltingpot_paintball_koth_mixed_70817_00000_0_2026-02-06_20-24-45/checkpoint_000193"

NUM_EPISODES = 5          # 실행할 에피소드 수
RENDER_SCALE = 5          # 화면 확대 배율 (MeltingPot 기본이 작으므로 확대 필요)
FPS = 15                  # 렌더링 속도 (초당 프레임 수)

# 정책 매핑 함수 (train.py와 동일하게 설정)
def policy_mapping_fn(agent_id, *args, **kwargs):
    if agent_id in ["player_0", "player_2"]:  # Red Team
        return "main_policy"
    else:  # Blue Team (player_1, player_3)
        return "opponent_policy"

# ==============================================================================
# [메인] 평가 루프
# ==============================================================================
def run_evaluation():
    # 1. Ray 초기화 및 모델/환경 등록
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)
    
    ModelCatalog.register_custom_model("meltingpot_model", MeltingPotModel)
    env_name = "meltingpot_paintball_koth_mixed"
    register_env(env_name, lambda cfg: env_creator({"substrate": "paintball__king_of_the_hill"}))

    print(f"🔄 Loading checkpoint from: {CHECKPOINT_PATH}")
    
    # 2. 알고리즘(체크포인트) 로드
    try:
        algo = Algorithm.from_checkpoint(CHECKPOINT_PATH)
    except Exception as e:
        print(f"❌ 체크포인트 로드 실패: {e}")
        print("경로가 정확한지, model.py와 env_utils.py가 같은 폴더에 있는지 확인해주세요.")
        return

    # 3. 환경 생성
    # 학습 때와 동일한 전처리가 포함된 환경을 생성합니다.
    env = env_creator({"substrate": "paintball__king_of_the_hill"})

    # 4. 에피소드 루프
    for i in range(NUM_EPISODES):
        print(f"\n🎬 Starting Episode {i+1}/{NUM_EPISODES}")
        
        obs, infos = env.reset()
        done = False
        
        # 점수 집계용
        episode_rewards = {agent_id: 0.0 for agent_id in env.par_env.possible_agents}
        step_count = 0

        # 에이전트 내부 상태 (모델이 Stateless여도 형식상 필요할 수 있음)
        # model.py를 보니 상태가 없는(Stateless) 모델이지만, 호환성을 위해 관리 구조만 유지
        agent_states = {} 

        while not done:
            # --- [렌더링] ---
            # Shimmy 환경은 par_env.render()를 통해 RGB 이미지를 얻을 수 있습니다.
            try:
                frame = env.par_env.render()
                if frame is not None:
                    # OpenCV는 BGR을 사용하므로 RGB -> BGR 변환
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    
                    # 화면 확대
                    h, w, _ = frame_bgr.shape
                    frame_resized = cv2.resize(frame_bgr, (w * RENDER_SCALE, h * RENDER_SCALE), interpolation=cv2.INTER_NEAREST)
                    
                    # 정보 텍스트 표시
                    cv2.putText(frame_resized, f"Ep {i+1} | Step {step_count}", (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
                    cv2.imshow("Melting Pot - Paintball KOTH", frame_resized)
                    
                    # 키 입력 대기 (ESC 누르면 종료)
                    if cv2.waitKey(1000 // FPS) & 0xFF == 27:
                        print("User interrupted.")
                        env.close()
                        cv2.destroyAllWindows()
                        return
            except Exception as e:
                print(f"Rendering error: {e}")

            # --- [행동 결정] ---
            actions = {}
            for agent_id, agent_obs in obs.items():
                policy_id = policy_mapping_fn(agent_id)
                
                # 상태 초기화 (필요시)
                if agent_id not in agent_states:
                    policy = algo.get_policy(policy_id)
                    agent_states[agent_id] = policy.get_initial_state()

                # 행동 계산
                # explore=False로 설정하여 결정론적(최적) 행동을 하도록 함
                compute_result = algo.compute_single_action(
                    agent_obs,
                    state=agent_states[agent_id],
                    policy_id=policy_id,
                    explore=False, 
                    full_fetch=True
                )
                
                # 결과 언패킹 (callbacks.py의 로직 참조)
                if isinstance(compute_result, tuple) and len(compute_result) >= 3:
                    action, state_out, _ = compute_result
                else:
                    action = compute_result
                    state_out = agent_states[agent_id]
                
                actions[agent_id] = action
                agent_states[agent_id] = state_out

            # --- [스텝 진행] ---
            obs, rewards, terminations, truncations, infos = env.step(actions)
            step_count += 1

            # 리워드 누적
            for agent_id, reward in rewards.items():
                episode_rewards[agent_id] += reward

            # 종료 조건 확인
            if any(terminations.values()) or all(truncations.values()) or len(obs) == 0:
                done = True

        # --- [에피소드 결과 출력] ---
        print(f"✅ Episode {i+1} Finished ({step_count} steps)")
        print("   [Scores]")
        
        red_score = episode_rewards.get("player_0", 0) + episode_rewards.get("player_2", 0)
        blue_score = episode_rewards.get("player_1", 0) + episode_rewards.get("player_3", 0)
        
        for agent_id, score in episode_rewards.items():
            team = "(Red)" if agent_id in ["player_0", "player_2"] else "(Blue)"
            print(f"   - {agent_id} {team}: {score:.2f}")
        
        print(f"   🏆 Result: Red {red_score:.1f} vs Blue {blue_score:.1f}")
        print("-" * 40)

    # 종료 처리
    env.close()
    cv2.destroyAllWindows()
    ray.shutdown()
    print("All evaluation episodes completed.")

if __name__ == "__main__":
    run_evaluation()