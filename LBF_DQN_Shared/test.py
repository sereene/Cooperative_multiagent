import argparse
import os
import cv2  # 시각화를 위해 OpenCV 필요 (pip install opencv-python)
import numpy as np
import ray
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env

# 기존 프로젝트 파일 임포트
from model import CustomMLP
from env_utils import env_creator

if not hasattr(np, 'product'):
    np.product = np.prod

def run_test(checkpoint_path, env_id, num_episodes=5, render=True, slow_mo=0.1):
    # 1. Ray 초기화
    if not ray.is_initialized():
        ray.init()

    # 2. 모델 및 환경 등록 (학습 때와 동일해야 함)
    ModelCatalog.register_custom_model("custom_mlp", CustomMLP)
    register_env("LBF_env", lambda cfg: env_creator(cfg))

    print(f"Loading checkpoint from: {checkpoint_path}")
    
    # 3. 체크포인트에서 알고리즘 복원
    # RLLib은 체크포인트 내의 config와 state를 자동으로 로드합니다.
    try:
        algo = Algorithm.from_checkpoint(checkpoint_path)
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        print("경로가 정확한지 확인해주세요 (예: .../checkpoint_000100)")
        return

    # 4. 테스트용 환경 생성
    env_config = {"env_id": env_id}
    env = env_creator(env_config)

    total_rewards = []
    success_rates = []

    for episode in range(num_episodes):
        obs, info = env.reset()
        done = False
        score = 0
        step = 0
        
        print(f"\n--- Episode {episode + 1} Start ---")

        while not done:
            if render:
                # LBF render returns RGB array
                frame = env.render()
                if frame is not None:
                    # OpenCV는 BGR을 사용하므로 RGB -> BGR 변환
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    # 화면 크기 키우기 (선택 사항)
                    frame = cv2.resize(frame, (400, 400), interpolation=cv2.INTER_NEAREST)
                    
                    cv2.imshow("LBF Agent Test", frame)
                    
                    # 키 입력 대기 (ESC 누르면 종료)
                    if cv2.waitKey(int(slow_mo * 1000)) & 0xFF == 27:
                        print("Test interrupted by user.")
                        env.close()
                        cv2.destroyAllWindows()
                        return

            # 행동 결정 (Exploration 끄기)
            actions = {}
            for agent_id, agent_obs in obs.items():
                action = algo.compute_single_action(
                    agent_obs,
                    policy_id="shared_policy",  # config에 설정된 정책 ID
                    explore=False               # 테스트 시에는 탐험 끔
                )
                actions[agent_id] = action

            # 환경 진행
            obs, rewards, terminations, truncations, infos = env.step(actions)
            
            # 점수 합산
            # 모든 에이전트가 협동 보상을 받으므로, 하나(agent_0)의 보상만 더하거나 평균을 냄
            # 여기서는 팀 전체 스코어를 보기 위해 agent_0 기준 합산 (Cooperative Pong 등은 보통 공유 보상)
            score += sum(rewards.values()) / len(rewards) # 평균 보상

            done = terminations["__all__"] or truncations["__all__"]
            step += 1

        # 에피소드 종료 후 메트릭 수집
        # env_utils.py에서 step의 마지막 infos에 success_rate를 넣도록 수정했으므로 가져옵니다.
        final_success_rate = 0.0
        if "agent_0" in infos and "success_rate" in infos["agent_0"]:
            final_success_rate = infos["agent_0"]["success_rate"]
        
        total_rewards.append(score)
        success_rates.append(final_success_rate)
        
        print(f"Episode {episode + 1} Done. Steps: {step}, Score: {score:.2f}, Success Rate: {final_success_rate*100:.1f}%")

    env.close()
    cv2.destroyAllWindows()
    ray.shutdown()

    # 최종 결과 출력
    print("\n" + "="*30)
    print(f"TEST RESULTS ({num_episodes} episodes)")
    print("="*30)
    print(f"Avg Score: {np.mean(total_rewards):.4f}")
    print(f"Avg Success Rate: {np.mean(success_rates)*100:.2f}%")
    print("="*30)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # 체크포인트 경로 인자 (필수)
    parser.add_argument("--checkpoint", type=str, required=True, 
                        help="Path to the checkpoint directory (e.g., results/.../checkpoint_000100)")
    
    # 환경 ID (기본값 설정됨)
    parser.add_argument("--env", type=str, default="Foraging-8x8-2p-2f-v3", 
                        help="LBF environment ID")
    
    # 렌더링 여부
    parser.add_argument("--no-render", action="store_true", help="Disable rendering")
    
    args = parser.parse_args()

    run_test(
        checkpoint_path=args.checkpoint,
        env_id=args.env,
        num_episodes=10,        # 테스트할 에피소드 수
        render=not args.no_render,
        slow_mo=0.05            # 프레임 간 딜레이 (초)
    )