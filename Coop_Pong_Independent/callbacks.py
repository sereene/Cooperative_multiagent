import os
import gc
import numpy as np
import imageio.v2 as imageio
from ray.rllib.algorithms.callbacks import DefaultCallbacks
import supersuit as ss
from pettingzoo.butterfly import cooperative_pong_v5
import wandb  # [추가] WandB 업로드를 위해 임포트

# 환경 생성 함수를 가져옵니다.
from env_utils import FixedParallelPettingZooEnv, env_creator, MAX_CYCLES
from RewardShapingWrapper import RewardShapingWrapper
# from MirrorObservationWrapper import MirrorObservationWrapper

class CoopPongCallbacks(DefaultCallbacks):
    def on_episode_end(self, *, worker, base_env, policies, episode, **kwargs):
        length = episode.length
        # MAX_CYCLES까지 버텼으면 성공으로 간주
        success = 1.0 if length >= MAX_CYCLES - 1 else 0.0
        episode.custom_metrics["success"] = success

def rollout_and_save_gif(
    *,
    algorithm,
    out_path: str,
    max_cycles: int,
    every_n_steps: int = 4,
    max_frames: int = 200,
    fps: int = 30,
):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # env_creator를 통해 환경 생성 (Wrapper가 적용된 상태여야 함)
    env = env_creator()

    frames = []
    try:
        obs, infos = env.reset()
        step_i = 0

        fr0 = env.render()
        if fr0 is not None: frames.append(fr0)

        terminations = {a: False for a in env.possible_agents}
        truncations = {a: False for a in env.possible_agents}

        while True:
            if all(terminations.get(a, False) or truncations.get(a, False) for a in env.possible_agents):
                break

            actions = {}
            for agent_id, agent_obs in obs.items():
                # agent_id에 맞는 policy를 사용하여 행동 결정
                action = algorithm.compute_single_action(agent_obs, policy_id=agent_id, explore=False)
                actions[agent_id] = action

            obs, rewards, terminations, truncations, infos = env.step(actions)

            if (step_i % every_n_steps) == 0:
                if len(frames) >= max_frames: break
                fr = env.render()
                if fr is not None: frames.append(fr)

            step_i += 1

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

        gif_filename = f"eval_{self.eval_count:04d}_iter{training_iter:06d}.gif"
        out_path = os.path.join(self.out_dir, gif_filename)
        
        # GIF 생성 및 로컬 저장
        rollout_and_save_gif(algorithm=algorithm, out_path=out_path, max_cycles=self.max_cycles)

        # [추가] WandB에 GIF 업로드
        if os.path.exists(out_path):
            try:
                # WandB Run이 활성화되어 있는지 확인
                if wandb.run is not None:
                    wandb.log({
                        "evaluation/gif": wandb.Video(out_path, fps=30, format="gif", caption=gif_filename),
                        "global_step": result.get("timesteps_total", 0)
                    })
                    print(f"[WandB] Uploaded GIF: {gif_filename}")
            except Exception as e:
                print(f"[Warning] WandB upload failed: {e}")