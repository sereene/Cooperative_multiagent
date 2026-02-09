import os
import gc
import numpy as np
import imageio.v2 as imageio
from ray.rllib.algorithms.callbacks import DefaultCallbacks
# 수동 생성에 필요한 라이브러리 제거
# import supersuit as ss 
# from pettingzoo.butterfly import knights_archers_zombies_v10

from RewardShapingWrapper import RewardShapingWrapper
from env_utils import FixedParallelPettingZooEnv, env_creator, MAX_CYCLES

class CoopPongCallbacks(DefaultCallbacks):
    def on_episode_end(self, *, worker, base_env, policies, episode, **kwargs):
        length = episode.length
        
        # 1. 성공 여부 (오래 버팀)
        success = 1.0 if length >= MAX_CYCLES * 0.9 else 0.0
        episode.custom_metrics["success"] = success

        # 2. 점수 (킬 수)
        total_score = episode.total_reward
        episode.custom_metrics["score"] = total_score
        
        # 3. 종합 점수 (선택 사항: 점수 + 생존 보너스)
        combined_score = total_score + (success * 100.0)
        episode.custom_metrics["combined_score"] = combined_score

def rollout_and_save_gif(
    *,
    algorithm,
    out_path: str,
    max_cycles: int,
    every_n_steps: int = 4,
    max_frames: int = 300,
    fps: int = 30,
):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # [수정] 수동 환경 생성 코드를 제거하고 env_creator를 사용하여 일관성 보장
    # 이렇게 하면 env_utils.py에서 설정한 FrameStack이나 에이전트 수(기사 2명 등)가 자동으로 반영됩니다.
    env = env_creator({})
    
    frames = []
    try:
        # env_creator가 반환하는 env는 FixedParallelPettingZooEnv(RLLib Wrapper)입니다.
        obs, infos = env.reset()
        step_i = 0
        
        # RLLib Wrapper는 render()를 직접 노출하지 않을 수 있으므로 par_env를 통해 호출
        # (env_utils의 FixedParallelPettingZooEnv 구조에 따라 접근)
        if hasattr(env, "par_env") and hasattr(env.par_env, "render"):
             fr0 = env.par_env.render()
        elif hasattr(env, "render"):
             fr0 = env.render()
        else:
             fr0 = None
        
        if fr0 is not None: frames.append(fr0)

        # Termination 확인을 위한 에이전트 목록 가져오기
        base_env = env.par_env if hasattr(env, "par_env") else env
        possible_agents = base_env.possible_agents if hasattr(base_env, "possible_agents") else obs.keys()

        terminations = {a: False for a in possible_agents}
        truncations = {a: False for a in possible_agents}

        while True:
            if not obs: break
            
            actions = {}
            for agent_id, agent_obs in obs.items():
                # 각 에이전트의 정책으로 행동 결정
                # 정책 ID 매핑이 복잡한 경우 algorithm.compute_single_action 호출 시 policy_id 지정 필요
                # 여기서는 agent_id와 policy_id가 동일하다고 가정 (policy_mapping_fn 참조)
                action = algorithm.compute_single_action(
                    agent_obs, 
                    policy_id=agent_id, 
                    explore=False
                )
                actions[agent_id] = action

            obs, rewards, terminations, truncations, infos = env.step(actions)

            if (step_i % every_n_steps) == 0:
                if len(frames) >= max_frames: break
                
                if hasattr(env, "par_env") and hasattr(env.par_env, "render"):
                    fr = env.par_env.render()
                elif hasattr(env, "render"):
                    fr = env.render()
                else:
                    fr = None
                    
                if fr is not None: frames.append(fr)

            step_i += 1
            
            # RLLib wrapper는 "__all__" 키를 포함합니다.
            if terminations.get("__all__", False) or truncations.get("__all__", False) or len(obs) == 0:
                break

        if frames:
            imageio.mimsave(out_path, frames, fps=fps)
            print(f"[GIF] saved: {out_path} frames={len(frames)}")
        else:
            print(f"[GIF] skipped (no frames): {out_path}")

    finally:
        try:
            env.close()
            gc.collect()
        except Exception:
            pass


class GifCallbacks(CoopPongCallbacks):
    def __init__(self, out_dir: str, every_n_evals: int = 5, max_cycles: int = 500):
        super().__init__()
        self.out_dir = out_dir
        self.every_n_evals = every_n_evals
        self.max_cycles = max_cycles
        self.eval_count = 0
        os.makedirs(self.out_dir, exist_ok=True)

    def on_train_result(self, *, algorithm, result, **kwargs):
        training_iter = int(result.get("training_iteration", 0))
        # evaluation 결과가 없으면 건너뜀 (train_batch_size 등에 따라 매번 evaluation이 안 돌 수도 있음)
        if "evaluation" not in result: return

        self.eval_count += 1
        if (self.eval_count % self.every_n_evals) != 0: return

        out_path = os.path.join(self.out_dir, f"eval_{self.eval_count:04d}_iter{training_iter:06d}.gif")
        rollout_and_save_gif(algorithm=algorithm, out_path=out_path, max_cycles=self.max_cycles)