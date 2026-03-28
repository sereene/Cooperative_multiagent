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
    env,
    max_cycles: int = 100,
    fps: int = 15,  
):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    frames = []

    try:
        # 매번 새로 만들지 않고 기존 환경을 reset하여 재사용 (OOM 방지)
        obs, infos = env.reset()
        step_i = 0
        
        # 비디오 저장을 위한 환경 Render 함수 찾기
        def get_frame():
            curr_env = env
            while hasattr(curr_env, "env"):
                if hasattr(curr_env, "render"):
                    try:
                        return curr_env.render(mode="rgb_array")
                    except Exception:
                        pass
                curr_env = curr_env.env
            
            if hasattr(curr_env, "render"):
                return curr_env.render(mode="rgb_array")
            return None

        fr0 = get_frame()
        if fr0 is not None:
            frames.append(fr0)

        # QMIX 기본 정책 및 초기 상태 설정
        policy_id = "default_policy"
        policy = algorithm.get_policy(policy_id)
        state = policy.get_initial_state() if hasattr(policy, "get_initial_state") else []

        while True:
            if step_i >= max_cycles:
                break
                
            actions = {}
            for agent_group_id, group_obs in obs.items():
                pass_state = state if len(state) > 0 else None
                
                result = algorithm.compute_single_action(
                    group_obs,
                    state=pass_state,
                    policy_id=policy_id,
                    explore=False,  # 평가 모드 (탐험 X)
                )
                
                if isinstance(result, tuple):
                    action, state_out, _ = result
                else:
                    action = result
                    state_out = []

                actions[agent_group_id] = action
                state = state_out  # RNN State 업데이트

            # 환경 스텝 진행
            obs, rewards, terminations, truncations, infos = env.step(actions)

            fr = get_frame()
            if fr is not None:
                frames.append(fr)

            step_i += 1

            if terminations.get("__all__", False) or truncations.get("__all__", False):
                break

        if not frames:
            print("[VIDEO] No frames captured.")
            return None

        safe_frames = []
        for fr in frames:
            if isinstance(fr, np.ndarray):
                if fr.dtype != np.uint8:
                    fr = np.clip(fr, 0, 255).astype(np.uint8)
                safe_frames.append(fr)

        with imageio.get_writer(
            out_path,
            fps=fps,
            codec="libx264",
            macro_block_size=None,
            ffmpeg_params=["-pix_fmt", "yuv420p"],
        ) as writer:
            for fr in safe_frames:
                writer.append_data(fr)

        print(f"[VIDEO] Saved: {out_path} ({len(safe_frames)} frames)")
        
        # WandB 업로드를 위해 저장된 파일의 경로를 반환합니다.
        return out_path

    finally:
        # 환경을 닫지 않고 다음 에피소드를 위해 프레임 리스트만 비웁니다.
        frames.clear()
        gc.collect()

class VideoCallbacks(DefaultCallbacks):
    def __init__(self, out_dir: str, env_creator_fn, every_n_evals: int = 5):
        super().__init__()
        self.base_out_dir = out_dir
        self.run_ts = _get_run_timestamp()
        
        self.run_dir = os.path.join(self.base_out_dir, self.run_ts)
        os.makedirs(self.run_dir, exist_ok=True)

        self.every_n_evals = every_n_evals
        self.eval_count = 0
        self._last_saved_iter = -1
        
        # 콜백 초기화 시점에 렌더링용 환경을 단 한 번만 생성 (OOM 방지 핵심)
        self.eval_env = env_creator_fn({})

    def on_train_result(self, *, algorithm, result, **kwargs):
        if "evaluation" not in result:
            return

        self.eval_count += 1
        if (self.eval_count % self.every_n_evals) != 0:
            return

        training_iter = int(result.get("training_iteration", 0))

        if training_iter == self._last_saved_iter:
            return
        self._last_saved_iter = training_iter

        video_filename = f"eval_capture_target_iter{training_iter:06d}.mp4"
        out_path = os.path.join(self.run_dir, video_filename)

        # 동영상을 저장하고 경로를 받아옵니다.
        saved_path = rollout_and_save_video(
            algorithm=algorithm,
            out_path=out_path,
            env=self.eval_env,
            max_cycles=100,
            fps=15, 
        )
        
        # [WandB 연동 핵심] 생성된 비디오 객체를 RLlib의 result 딕셔너리에 직접 주입합니다.
        if saved_path and os.path.exists(saved_path):
            result["evaluation/gameplay"] = wandb.Video(saved_path, fps=15, format="mp4")
            
    def __del__(self):
        # 학습이 끝나고 프로세스가 종료될 때만 환경을 닫아 리소스를 반환합니다.
        if hasattr(self, 'eval_env'):
            try:
                self.eval_env.close()
            except Exception:
                pass