import os
import gc
import numpy as np
import imageio.v2 as imageio
from datetime import datetime
from ray.rllib.algorithms.callbacks import DefaultCallbacks
import wandb

# 환경 생성 함수를 가져옵니다.
from env_utils import env_creator, MAX_CYCLES

class CoopPongCallbacks(DefaultCallbacks):
    def on_episode_end(self, *, worker, base_env, policies, episode, **kwargs):
        length = episode.length
        success = 1.0 if length >= MAX_CYCLES - 1 else 0.0
        episode.custom_metrics["success"] = success

def rollout_and_save_gif(
    *,
    algorithm,
    out_path: str,
    max_cycles: int,
    every_n_steps: int = 4,
    max_frames: int = 200,
    fps: int = 15,
):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

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
            if "paddle_0" in obs and "paddle_1" in obs:
                grouped_obs = (obs["paddle_0"], obs["paddle_1"])
                
                # [수정] compute_single_action 반환값 처리 강화
                # 반환값이 (action, state, info) 튜플일 수도 있고, action만 반환될 수도 있습니다.
                result = algorithm.compute_single_action(
                    grouped_obs, 
                    state=[],        # 빈 리스트 전달 (Stateless)
                    policy_id="group_1", 
                    explore=False
                )
                
                # 반환값 구조 분해 (안전하게 처리)
                if isinstance(result, tuple) and len(result) >= 1:
                    group_action = result[0] # 첫 번째 요소가 action
                else:
                    group_action = result

                # group_action이 튜플/리스트인지 스칼라인지 확인 후 할당
                if isinstance(group_action, (list, tuple, np.ndarray)) and len(np.shape(group_action)) > 0:
                     # 배열 형태인 경우 (정상 케이스)
                    actions["paddle_0"] = group_action[0]
                    actions["paddle_1"] = group_action[1]
                else:
                    # [예외 처리] 만약 스칼라 값이 반환된다면 (단일 값인 경우)
                    # QMIX 그룹이지만 단일 값으로 나왔을 때를 대비해 로그 출력 후 처리
                    # 보통 이런 경우 에러가 발생할 수 있으므로 확인용 print 추가
                    # print(f"[DEBUG] group_action is scalar: {group_action}, type: {type(group_action)}")
                    # 비상용: 동일 액션 적용 혹은 0번 인덱스로 간주 (상황에 따라 다름)
                    try:
                        actions["paddle_0"] = group_action
                        actions["paddle_1"] = group_action # 혹은 다른 기본값
                    except Exception:
                        pass # 무시하거나 로그

            obs, rewards, terminations, truncations, infos = env.step(actions)

            if (step_i % every_n_steps) == 0:
                if len(frames) >= max_frames: break
                fr = env.render()
                if fr is not None: frames.append(fr)

            step_i += 1

        if frames:
            imageio.mimsave(out_path, frames, fps=fps)
            print(f"[GIF] saved locally: {out_path} ({len(frames)} frames)")
            
            if wandb.run is not None:
                try:
                    video_array = np.array(frames)
                    video_array = np.transpose(video_array, (0, 3, 1, 2))
                    
                    wandb.log({
                        "evaluation/gameplay_video": wandb.Video(
                            video_array, 
                            fps=fps, 
                            format="gif", 
                            caption=f"Eval Video: {os.path.basename(out_path)}"
                        ),
                        "global_step": algorithm.training_iteration
                    })
                except Exception as e:
                    print(f"[WandB] Upload failed: {e}")

    finally:
        try:
            env.close()
            gc.collect()
        except Exception:
            pass

class GifCallbacks(CoopPongCallbacks):
    def __init__(self, out_dir: str, every_n_evals: int = 5, max_cycles: int = 500):
        super().__init__()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.out_dir = os.path.join(out_dir, timestamp)
        self.every_n_evals = every_n_evals
        self.max_cycles = max_cycles
        self.eval_count = 0
        os.makedirs(self.out_dir, exist_ok=True)

    def on_train_result(self, *, algorithm, result, **kwargs):
        if "evaluation" not in result: return

        self.eval_count += 1
        if (self.eval_count % self.every_n_evals) != 0: return

        training_iter = int(result.get("training_iteration", 0))
        gif_filename = f"eval_{self.eval_count:04d}_iter{training_iter:06d}.gif"
        out_path = os.path.join(self.out_dir, gif_filename)
        
        rollout_and_save_gif(
            algorithm=algorithm, 
            out_path=out_path, 
            max_cycles=self.max_cycles,
            fps=15
        )