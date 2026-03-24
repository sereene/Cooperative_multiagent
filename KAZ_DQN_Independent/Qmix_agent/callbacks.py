import os
import gc
import numpy as np
import imageio.v2 as imageio
from datetime import datetime
from ray.rllib.algorithms.callbacks import DefaultCallbacks
import wandb

RUN_TS_ENV = "VIDEO_RUN_TS"

def _get_run_timestamp() -> str:
    ts = os.environ.get(RUN_TS_ENV)
    if not ts:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.environ[RUN_TS_ENV] = ts
    return ts

def rollout_and_save_video(
    *,
    algorithm,
    out_path: str,
    env_creator_fn,
    max_cycles: int = 900,
    fps: int = 30,  # KAZ는 속도감이 있으므로 30fps 추천
):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # 평가용 환경 생성 (QMIX용 grouped_env_creator 호출)
    env = env_creator_fn({})
    frames = []

    try:
        obs, infos = env.reset()
        step_i = 0

        # QMIX는 'default_policy' 하나만 사용합니다.
        policy = algorithm.get_policy("default_policy")
        
        # QMIX 내부의 RNN(GRU) 상태 초기화
        state = policy.get_initial_state() if hasattr(policy, "get_initial_state") else []

        def get_frame():
            # 래퍼 내부 깊숙이 있는 실제 환경의 render() 함수를 찾습니다.
            target_env = env
            while hasattr(target_env, "env") or hasattr(target_env, "par_env"):
                if hasattr(target_env, "render") and callable(target_env.render):
                    try:
                        # 렌더링 시도
                        fr = target_env.render()
                        if fr is not None:
                            return fr
                    except:
                        pass
                
                if hasattr(target_env, "par_env"):
                    target_env = target_env.par_env
                elif hasattr(target_env, "env"):
                    target_env = target_env.env
                else:
                    break
                    
            if hasattr(target_env, "render"):
                return target_env.render()
            return None

        fr0 = get_frame()
        if fr0 is not None:
            frames.append(fr0)

        while True:
            if step_i >= max_cycles:
                break

            actions = {}
            
            # QMIX는 "group_1" 이라는 그룹 단위로 묶인 관측값을 받아서 추론합니다.
            for agent_id, agent_obs in obs.items():
                pass_state = state if len(state) > 0 else None
                
                result = algorithm.compute_single_action(
                    agent_obs,
                    state=pass_state,
                    policy_id="default_policy",
                    explore=False,
                )
                
                if isinstance(result, tuple):
                    action, state_out, _ = result
                    state = state_out  # 상태 업데이트
                else:
                    action = result
                    
                actions[agent_id] = action

            obs, rewards, terminations, truncations, infos = env.step(actions)

            fr = get_frame()
            if fr is not None:
                frames.append(fr)

            step_i += 1

            if terminations.get("__all__", False) or truncations.get("__all__", False) or len(obs) == 0:
                break

        if not frames:
            print("[VIDEO] No frames captured (env.render() returned None).")
            return

        # MP4 비디오 로컬 저장 전처리
        safe_frames = []
        for fr in frames:
            if fr is None:
                continue
            if isinstance(fr, np.ndarray):
                if fr.dtype != np.uint8:
                    fr = np.clip(fr, 0, 255).astype(np.uint8)
                
                # 👇 libx264 코덱 에러 방지용 해상도 짝수 보정
                h, w, c = fr.shape
                if h % 2 != 0:
                    fr = fr[:-1, :, :]
                if w % 2 != 0:
                    fr = fr[:, :-1, :]
                    
                safe_frames.append(fr)

        if not safe_frames:
            print("[VIDEO] Frames existed but none were valid numpy arrays.")
            return

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

        # WandB 업로드
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
                    commit=False,
                )
            except Exception as e:
                print(f"[WandB] Video upload failed: {e}")

    finally:
        try:
            env.close()
        except Exception:
            pass
        gc.collect()

class VideoCallbacks(DefaultCallbacks):
    def __init__(self, out_dir: str, env_creator_fn, every_n_evals: int = 5, max_cycles: int = 900):
        super().__init__()
        self.base_out_dir = out_dir
        self.run_ts = _get_run_timestamp()
        
        self.run_dir = os.path.join(self.base_out_dir, self.run_ts)
        os.makedirs(self.run_dir, exist_ok=True)

        self.env_creator_fn = env_creator_fn
        self.every_n_evals = every_n_evals
        self.max_cycles = max_cycles
        self.eval_count = 0
        self._last_saved_iter = -1

    def on_train_result(self, *, algorithm, result, **kwargs):
        if "evaluation" not in result:
            return

        self.eval_count += 1
        # 설정한 주기마다 한 번씩 영상 생성
        if (self.eval_count % self.every_n_evals) != 0:
            return

        training_iter = int(result.get("training_iteration", 0))

        if training_iter == self._last_saved_iter:
            return
        self._last_saved_iter = training_iter

        video_filename = f"eval_{self.eval_count:04d}_iter{training_iter:06d}.mp4"
        out_path = os.path.join(self.run_dir, video_filename)

        rollout_and_save_video(
            algorithm=algorithm,
            out_path=out_path,
            env_creator_fn=self.env_creator_fn,
            max_cycles=self.max_cycles,
            fps=30,  # KAZ 환경 속도
        )