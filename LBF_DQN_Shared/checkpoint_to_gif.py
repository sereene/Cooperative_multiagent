import argparse
import os
import numpy as np
import ray
import imageio.v2 as imageio  # GIF 생성용
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env

# 기존 프로젝트 파일 임포트
from model import CustomMLP
from env_utils import env_creator

def save_gif(checkpoint_path, output_filename, env_id, fps=5, max_steps=100):
    # 1. Ray 초기화
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)

    # 2. 모델 및 환경 등록 (필수: 체크포인트 로드 전 수행)
    ModelCatalog.register_custom_model("custom_mlp", CustomMLP)
    register_env("LBF_env", lambda cfg: env_creator(cfg))

    print(f"Loading checkpoint from: {checkpoint_path}")
    
    # 3. 알고리즘 로드
    try:
        algo = Algorithm.from_checkpoint(checkpoint_path)
    except Exception as e:
        print(f"[Error] Failed to load checkpoint: {e}")
        return

    # 4. 환경 생성
    env_config = {"env_id": env_id}
    env = env_creator(env_config)

    frames = []
    
    # 에피소드 시작
    obs, info = env.reset()
    done = False
    step = 0
    
    # 첫 프레임 캡처
    init_frame = env.render()
    if init_frame is not None:
        frames.append(init_frame)

    print("Generating frames...")
    
    while not done and step < max_steps:
        actions = {}
        for agent_id, agent_obs in obs.items():
            # 학습된 정책으로 행동 결정 (탐험 X)
            action = algo.compute_single_action(
                agent_obs,
                policy_id="shared_policy",
                explore=False
            )
            actions[agent_id] = action

        # 환경 진행
        obs, rewards, terminations, truncations, infos = env.step(actions)
        
        # 프레임 캡처
        frame = env.render()
        if frame is not None:
            frames.append(frame)

        done = terminations["__all__"] or truncations["__all__"]
        step += 1

    env.close()
    ray.shutdown()

    # 5. GIF 저장
    if len(frames) > 0:
        # 출력 경로가 없으면 생성
        os.makedirs(os.path.dirname(output_filename), exist_ok=True)
        
        # GIF 저장 (loop=0은 무한 반복)
        imageio.mimsave(output_filename, frames, fps=fps, loop=0)
        print(f"\n[Success] GIF saved to: {output_filename}")
        print(f"Total Frames: {len(frames)}")
    else:
        print("[Error] No frames captured.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # 체크포인트 경로 (필수)
    parser.add_argument("--checkpoint", type=str, required=True, 
                        help="Path to the checkpoint directory")
    
    # 저장할 파일명
    parser.add_argument("--output", type=str, default="replay.gif", 
                        help="Output GIF filename")
    
    # 환경 ID
    parser.add_argument("--env", type=str, default="Foraging-8x8-2p-2f-v3", 
                        help="LBF environment ID")
    
    # FPS (속도 조절)
    parser.add_argument("--fps", type=int, default=5, 
                        help="Frames per second (Default: 5)")

    args = parser.parse_args()

    # 출력 경로가 상대경로면 현재 폴더 기준 절대경로로 변환 (저장 위치 명확히 하기 위함)
    output_path = os.path.abspath(args.output)

    save_gif(
        checkpoint_path=args.checkpoint,
        output_filename=output_path,
        env_id=args.env,
        fps=args.fps
    )