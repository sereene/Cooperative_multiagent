import os
import numpy as np
import cv2
import shimmy
import tensorflow as tf
import dm_env  # [핵심] TimeStep 객체 생성을 위해 필요

# Melting Pot의 정책 로드 유틸리티
from meltingpot_repo.meltingpot.utils.policies import saved_model_policy
from meltingpot_repo.meltingpot.utils.policies import policy

# ==============================================================================
# 1. 설정 (경로를 본인 환경에 맞게 확인하세요)
# ==============================================================================
SUBSTRATE_NAME = "commons_harvest__partnership"
RENDER_MODE = "human"
FPS = 8
SCALE_FACTOR = 8  # 화면 확대 배율 (Melting Pot 기본 화면이 작으므로 확대 추천)

# [사용자 경로 설정]
# 주의: 폴더 경로 끝에 /saved_model.pb 파일명을 포함하지 마세요. 폴더 경로만 입력합니다.
GOOD_PARTNER_MODEL_DIR = "/home/jsr/project/Cooperative_pong_RL_agent/Meltingpot_muliagent_env/meltingpot_repo/assets/saved_models/commons_harvest__partnership/sustainable_fighter_0"
INVADER_MODEL_DIR = "/home/jsr/project/Cooperative_pong_RL_agent/Meltingpot_muliagent_env/meltingpot_repo/assets/saved_models/commons_harvest__partnership/free_2"

# ==============================================================================
# 2. 유틸리티 함수 및 클래스
# ==============================================================================
def load_saved_model(path: str) -> policy.Policy:
    if os.path.exists(path):
        print(f"[로드 성공] {path}")
        return saved_model_policy.SavedModelPolicy(path)
    else:
        print(f"[경고] 모델을 찾을 수 없습니다. (랜덤 정책 사용): {path}")
        return RandomPolicy()

class RandomPolicy(policy.Policy):
    def step(self, observation, prev_state):
        return np.random.randint(0, 8), prev_state
    def initial_state(self): return ()
    def close(self): pass

# ==============================================================================
# 3. 메인 실행 함수
# ==============================================================================
def main():
    print(f"Shimmy를 통해 환경 생성 중: {SUBSTRATE_NAME}")
    
    # [1] Shimmy Wrapper로 환경 생성
    env = shimmy.MeltingPotCompatibilityV0(
        substrate_name=SUBSTRATE_NAME,
        render_mode=RENDER_MODE
    )

    print(f"Agents: {env.possible_agents}")

    # [2] 정책 로드 및 할당
    policies = {}
    policy_states = {}

    for agent_id in env.possible_agents:
        # agent_id 예시: 'player_0', 'player_1' ...
        idx = int(agent_id.split("_")[1])
        
        if idx < 2:  # Player 0, 1 (주인공: 파트너)
            policies[agent_id] = load_saved_model(GOOD_PARTNER_MODEL_DIR)
        else:        # Player 2~6 (적: 침입자)
            policies[agent_id] = load_saved_model(INVADER_MODEL_DIR)
            
        # 초기 내부 상태(RNN state) 초기화
        policy_states[agent_id] = policies[agent_id].initial_state()

    # [3] 시뮬레이션 루프
    observations, infos = env.reset()
    
    try:
        while env.agents:
            # --- 행동 결정 단계 ---
            actions = {}
            for agent_id in env.agents:
                obs_dict = observations[agent_id]
                
                # [핵심 수정 1] Dict -> dm_env.TimeStep 변환
                # SavedModelPolicy는 딕셔너리가 아닌 dm_env.TimeStep 객체를 원합니다.
                # 따라서 가짜(Dummy) TimeStep을 만들어 감싸줍니다.
                dummy_timestep = dm_env.TimeStep(
                    step_type=dm_env.StepType.MID, # 중간 단계로 가정
                    reward=0.0,                    # 추론 시 보상은 불필요 (0.0)
                    discount=1.0,
                    observation=obs_dict           # 실제 관측 데이터 (RGB 등)
                )
                
                # [핵심 수정 2] 정책 실행
                # policies는 딕셔너리이므로 append()가 아니라 키값 접근([])을 해야 합니다.
                action, next_state = policies[agent_id].step(dummy_timestep, policy_states[agent_id])
                
                actions[agent_id] = action
                policy_states[agent_id] = next_state

            # --- 환경 진행 단계 ---
            observations, rewards, terminations, truncations, infos = env.step(actions)

            
            # --- 종료 조건 확인 ---
            if not env.agents: # 모든 에이전트 종료 시
                print("에피소드 종료. 재시작...")
                observations, infos = env.reset()
                for agent_id in env.possible_agents:
                    policy_states[agent_id] = policies[agent_id].initial_state()

    except KeyboardInterrupt:
        print("\n강제 종료됨.")
    finally:
        env.close()
        cv2.destroyAllWindows()
        for p in policies.values():
            p.close()

if __name__ == "__main__":
    main()