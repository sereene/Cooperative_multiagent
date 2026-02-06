import numpy as np
import matplotlib.pyplot as plt
from env_utils import env_creator

def test_observation_shape():
    print("----------------------------------------------------------------")
    print("🧪 [테스트 시작] 환경 생성 및 관측값(FrameStack) 검증")
    print("----------------------------------------------------------------")

    # 1. 환경 생성
    # env_utils.py에서 FrameStackWrapper(num_stack=4)가 적용되어 있어야 함
    env_config = {"substrate": "paintball__king_of_the_hill"}
    env = env_creator(env_config)
    
    # [수정] Wrapper 종류에 따라 에이전트 목록을 가져오는 위치가 다름
    if hasattr(env, "possible_agents"):
        agents = env.possible_agents
    else:
        # FixedParallelPettingZooEnv 같은 래퍼는 par_env 안에 원본이 있음
        agents = env.par_env.possible_agents

    print(f"✅ 환경 생성 완료: {env}")
    print(f"👥 에이전트 목록: {agents}")

    # 2. Reset 테스트
    print("\n🔄 환경 Reset 중...")
    obs, infos = env.reset()

    # 3. 관측값 형태(Shape) 확인
    target_agent = agents[0] # 첫 번째 에이전트
    agent_obs = obs[target_agent]

    print(f"\n🕵️ [검증 결과] 에이전트 '{target_agent}'의 관측값:")
    print(f"   ▶ 타입: {type(agent_obs)}")
    
    if isinstance(agent_obs, np.ndarray):
        print(f"   ▶ Shape: {agent_obs.shape}")
        
        # 4. 검증 로직
        # 예상: (88, 88, 9) 또는 (88, 88, 12) (FrameStack 설정에 따라 다름)
        # env_utils.py에서 num_stack=3이면 9채널, 4면 12채널
        # 현재 코드에서는 num_stack=3으로 되어 있으므로 9가 나올 것임
        c_dim = agent_obs.shape[-1]
        
        if c_dim in [9, 12]:
            print(f"   ✅ 성공! 채널 수가 {c_dim}개입니다. (3채널 x {c_dim//3}프레임)")
        else:
            print(f"   ❌ 주의! 예상치 못한 채널 수: {c_dim}")
            print("      (env_utils.py의 FrameStackWrapper 설정을 확인하세요)")
    else:
        print("   ❌ 에러: 관측값이 Numpy 배열이 아닙니다. (Wrapper 문제 가능성)")
        print(f"      실제 값: {agent_obs}")

    # 5. 시각화 (선택 사항)
    if isinstance(agent_obs, np.ndarray) and agent_obs.shape[-1] >= 3:
        print("\n🖼️ 프레임 스택 시각화 (obs_test.png로 저장)")
        
        num_stack = agent_obs.shape[-1] // 3
        fig, axes = plt.subplots(1, num_stack, figsize=(4 * num_stack, 4))
        
        # 1장일 경우 axes가 배열이 아니므로 리스트로 변환
        if num_stack == 1: axes = [axes]

        # 3채널씩(RGB) 끊어서 복원
        for i in range(num_stack):
            # FrameStack은 [Oldest ... Newest] 순서로 쌓임
            start = i * 3
            end = (i + 1) * 3
            img_slice = agent_obs[:, :, start:end]
            
            axes[i].imshow(img_slice.astype(np.uint8))
            axes[i].set_title(f"Frame -{num_stack-1-i} (Ch {start}~{end-1})")
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig("obs_test.png")
        print("   ✅ 시각화 파일 저장 완료: obs_test.png")

    env.close()
    print("\n----------------------------------------------------------------")
    print("🏁 테스트 종료")
    print("----------------------------------------------------------------")

if __name__ == "__main__":
    test_observation_shape()