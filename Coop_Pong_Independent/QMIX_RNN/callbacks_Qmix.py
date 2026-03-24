import os
import gc
import numpy as np
import imageio.v2 as imageio
from datetime import datetime
from ray.rllib.algorithms.callbacks import DefaultCallbacks
import wandb

# 환경 생성 함수를 가져옵니다.
from env_utils import env_creator, MAX_CYCLES

# -----------------------------------------------------------------------------
# Run-level timestamp: set this ONCE in the training entrypoint (recommended).
# If not set, we set it once per-process as a fallback.
# -----------------------------------------------------------------------------
RUN_TS_ENV = "VIDEO_RUN_TS"

def _get_run_timestamp() -> str:
    ts = os.environ.get(RUN_TS_ENV)
    if not ts:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.environ[RUN_TS_ENV] = ts
    return ts


class CoopPongCallbacks(DefaultCallbacks):
    def on_episode_end(self, *, worker, base_env, policies, episode, **kwargs):
        length = episode.length
        success = 1.0 if length >= MAX_CYCLES - 1 else 0.0
        episode.custom_metrics["success"] = success


def rollout_and_save_video(
    *,
    algorithm,
    out_path: str,
    max_cycles: int,
    every_n_steps: int = 4,
    max_frames: int = 300,
    fps: int = 15,
):
    """Rollout a short greedy episode and save it as an MP4.

    Notes
    - Uses env_creator() which is already configured with render_mode="rgb_array".
    - Uses algorithm.compute_single_action(..., explore=False) so it is deterministic.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    env = env_creator()
    frames = []

    try:
        obs, infos = env.reset()
        step_i = 0

        # [중요] RNN 초기 상태
        rnn_state = algorithm.get_policy("group_1").get_initial_state()

        fr0 = env.render()
        if fr0 is not None:
            frames.append(fr0)

        terminations = {a: False for a in env.possible_agents}
        truncations = {a: False for a in env.possible_agents}

        while True:
            if step_i >= max_cycles:
                break

            if all(terminations.get(a, False) or truncations.get(a, False) for a in env.possible_agents):
                break

            actions = {}
            if "paddle_0" in obs and "paddle_1" in obs:
                grouped_obs = (obs["paddle_0"], obs["paddle_1"])

                group_action, rnn_state, _ = algorithm.compute_single_action(
                    grouped_obs,
                    state=rnn_state,
                    policy_id="group_1",
                    explore=False,
                )

                actions["paddle_0"] = group_action[0]
                actions["paddle_1"] = group_action[1]

            obs, rewards, terminations, truncations, infos = env.step(actions)

            if (step_i % every_n_steps) == 0:
                if len(frames) >= max_frames:
                    break
                fr = env.render()
                if fr is not None:
                    frames.append(fr)

            step_i += 1

        if not frames:
            print("[VIDEO] No frames captured (env.render() returned None).")
            return

        # -------------------------
        # Save MP4 locally
        # -------------------------
        # Ensure uint8 frames (H, W, C)
        safe_frames = []
        for fr in frames:
            if fr is None:
                continue
            if isinstance(fr, np.ndarray):
                if fr.dtype != np.uint8:
                    fr = np.clip(fr, 0, 255).astype(np.uint8)
                safe_frames.append(fr)

        if not safe_frames:
            print("[VIDEO] Frames existed but none were valid numpy arrays.")
            return

        # ffmpeg requires dimensions multiple of macro blocks in some cases.
        # macro_block_size=None disables this restriction (imageio will handle it).
        with imageio.get_writer(
            out_path,
            fps=fps,
            codec="libx264",
            macro_block_size=None,
            ffmpeg_params=["-pix_fmt", "yuv420p"],
        ) as writer:
            for fr in safe_frames:
                writer.append_data(fr)

        print(f"[VIDEO] saved locally: {out_path} ({len(safe_frames)} frames)")

        # -------------------------
        # Upload to W&B (if active)
        # -------------------------
        if wandb.run is not None:
            try:
                wandb.log(
                    {
                        "evaluation/gameplay_video": wandb.Video(
                            out_path,
                            fps=fps,
                            format="mp4",
                            caption=f"Eval iter={algorithm.training_iteration}",
                        )
                    },
                    step=int(algorithm.training_iteration),
                    commit=True,
                )
            except Exception as e:
                print(f"[WandB] Video upload failed: {e}")

    finally:
        try:
            env.close()
        except Exception:
            pass
        gc.collect()


class VideoCallbacks(CoopPongCallbacks):
    """Create ONE run folder per training launch and keep accumulating MP4s inside it."""

    def __init__(self, out_dir: str, every_n_evals: int = 5, max_cycles: int = 500):
        super().__init__()
        self.base_out_dir = out_dir
        self.run_ts = _get_run_timestamp()
        self.run_dir = os.path.join(self.base_out_dir, self.run_ts)
        os.makedirs(self.run_dir, exist_ok=True)

        self.every_n_evals = every_n_evals
        self.max_cycles = max_cycles
        self.eval_count = 0

        # Avoid duplicate saves if multiple callback instances exist in the same process.
        self._last_saved_iter = -1

    def on_train_result(self, *, algorithm, result, **kwargs):
        if "evaluation" not in result:
            return

        self.eval_count += 1
        if (self.eval_count % self.every_n_evals) != 0:
            return

        training_iter = int(result.get("training_iteration", 0))

        # If something calls this twice for the same iteration, skip duplicates.
        if training_iter == self._last_saved_iter:
            return
        self._last_saved_iter = training_iter

        video_filename = f"eval_{self.eval_count:04d}_iter{training_iter:06d}.mp4"
        out_path = os.path.join(self.run_dir, video_filename)

        rollout_and_save_video(
            algorithm=algorithm,
            out_path=out_path,
            max_cycles=self.max_cycles,
            fps=15,
        )


# Backward-compatible name (so your train script can keep importing GifCallbacks)
GifCallbacks = VideoCallbacks