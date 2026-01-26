import os
import gc
import numpy as np
import imageio.v2 as imageio
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from env_utils import env_creator

class LBFCallbacks(DefaultCallbacks):
    def on_episode_end(self, *, worker, base_env, policies, episode, **kwargs):
        # 1. 기본 점수 (Total Reward)
        total_score = episode.total_reward
        episode.custom_metrics["score"] = total_score
        
        # 2. 성공률 (Success Rate) 추가
        # 마지막 스텝의 info에서 success_rate를 가져옵니다.
        # 모든 에이전트가 동일한 success_rate를 공유하므로 agent_0의 info를 확인합니다.
        last_info = episode.last_info_for("agent_0")
        
        if last_info and "success_rate" in last_info:
            episode.custom_metrics["success_rate"] = last_info["success_rate"]


def rollout_and_save_gif(
    *,
    algorithm,
    out_path: str,
    env_name: str,
    max_steps: int = 50,
):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    env_config = {"env_id": env_name}
    env = env_creator(env_config)
    
    frames = []
    try:
        obs, _ = env.reset()
        
        fr0 = env.render()
        if fr0 is not None: frames.append(fr0)

        for _ in range(max_steps):
            if not obs: break
            
            actions = {}
            for agent_id, agent_obs in obs.items():
                action = algorithm.compute_single_action(
                    agent_obs, 
                    policy_id="shared_policy",
                    explore=False
                )
                actions[agent_id] = action

            obs, rewards, terminations, truncations, infos = env.step(actions)
            
            fr = env.render()
            if fr is not None: frames.append(fr)
            
            if terminations["__all__"] or truncations["__all__"]:
                break

        if frames:
            imageio.mimsave(out_path, frames, fps=5)
            print(f"[GIF] saved: {out_path} frames={len(frames)}")
        else:
            print(f"[GIF] skipped (no frames): {out_path}")

    finally:
        try:
            env.close()
            gc.collect()
        except Exception:
            pass

class GifCallbacks(LBFCallbacks):
    def __init__(self, out_dir: str, env_name: str, every_n_evals: int = 5):
        super().__init__()
        self.out_dir = out_dir
        self.env_name = env_name
        self.every_n_evals = every_n_evals
        self.eval_count = 0
        os.makedirs(self.out_dir, exist_ok=True)

    def on_train_result(self, *, algorithm, result, **kwargs):
        training_iter = int(result.get("training_iteration", 0))
        if "evaluation" not in result: return

        self.eval_count += 1
        if (self.eval_count % self.every_n_evals) != 0: return

        out_path = os.path.join(self.out_dir, f"eval_{self.eval_count:04d}_iter{training_iter:06d}.gif")
        rollout_and_save_gif(
            algorithm=algorithm, 
            out_path=out_path, 
            env_name=self.env_name
        )