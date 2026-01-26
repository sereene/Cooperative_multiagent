import os
import gc
import numpy as np
import imageio.v2 as imageio
from ray.rllib.algorithms.callbacks import DefaultCallbacks
import supersuit as ss
from pettingzoo.butterfly import knights_archers_zombies_v10

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
        # 생존 시 100점의 가치를 둠
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

    # [중요] 평가용 환경도 학습 환경과 똑같이 설정 (1 archer, 1 knight, spawn 50)
    env = knights_archers_zombies_v10.parallel_env(
        spawn_rate=50, 
        num_archers=1, 
        num_knights=1,
        max_arrows=1,
        max_cycles=max_cycles, 
        vector_state=True,
        render_mode="rgb_array"
    )
    # env = ss.frame_stack_v1(env, 4)

    env = RewardShapingWrapper(env)
    
    env = FixedParallelPettingZooEnv(env)
    
    frames = []
    try:
        obs, infos = env.reset()
        step_i = 0
        
        fr0 = env.render()
        if fr0 is not None: frames.append(fr0)

        # FixedParallelPettingZooEnv는 RLLib 래퍼라서 직접 속성이 없으므로,
        # 내부의 par_env(PettingZoo 환경)를 통해 접근해야 합니다.
        terminations = {a: False for a in env.par_env.possible_agents}
        truncations = {a: False for a in env.par_env.possible_agents}

        while True:
            if not obs: break
            
            actions = {}
            for agent_id, agent_obs in obs.items():
                # 각 에이전트의 정책으로 행동 결정
                action = algorithm.compute_single_action(
                    agent_obs, 
                    policy_id=agent_id, 
                    explore=False
                )
                actions[agent_id] = action

            obs, rewards, terminations, truncations, infos = env.step(actions)

            if (step_i % every_n_steps) == 0:
                if len(frames) >= max_frames: break
                fr = env.render()
                if fr is not None: frames.append(fr)

            step_i += 1
            
            if any(terminations.values()) or all(truncations.values()) or len(obs) == 0:
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
        if "evaluation" not in result: return

        self.eval_count += 1
        if (self.eval_count % self.every_n_evals) != 0: return

        out_path = os.path.join(self.out_dir, f"eval_{self.eval_count:04d}_iter{training_iter:06d}.gif")
        rollout_and_save_gif(algorithm=algorithm, out_path=out_path, max_cycles=self.max_cycles)