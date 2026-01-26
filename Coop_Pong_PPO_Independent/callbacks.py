import os
import gc
import imageio.v2 as imageio
from ray.rllib.algorithms.callbacks import DefaultCallbacks
import supersuit as ss
from pettingzoo.butterfly import cooperative_pong_v5

# env_creator는 내부에서 따로 생성하므로 상수만 가져오거나 직접 구현
from env_utils import MAX_CYCLES
from Coop_Pong_DQN_Independent.RewardShapingWrapper import RewardShapingWrapper
from Coop_Pong_DQN_Independent.MirrorObservationWrapper import MirrorObservationWrapper

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
    fps: int = 30,
):
    """
    algorithm(현재 학습 중인 PPO/Algo 인스턴스)을 이용해
    cooperative_pong_v5에서 1 에피소드 롤아웃하며 rgb_array 렌더 프레임을 모아 GIF 저장.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    env = cooperative_pong_v5.parallel_env(
        max_cycles=max_cycles,
        render_mode="rgb_array",
    )

    # 1. 크기 줄이기 (AI에게는 84x84로 보임, 하지만 render() 결과는 고화질일 수 있음)
    env = ss.resize_v1(env, x_size=84, y_size=168)

    # 2. 흑백 변환
    # env = ss.color_reduction_v0(env, mode='full')
    
    # 3. 프레임 스택 (4장 겹치기)
    # env = ss.frame_stack_v1(env, 3)
    
    env = RewardShapingWrapper(env)

    env = MirrorObservationWrapper(env)

    frames = []
    try:
        obs, infos = env.reset()
        step_i = 0

        # 첫 프레임
        fr0 = env.render()
        if fr0 is not None:
            frames.append(fr0)

        terminations = {a: False for a in env.possible_agents}
        truncations = {a: False for a in env.possible_agents}

        while True:
            if all(
                terminations.get(a, False) or truncations.get(a, False)
                for a in env.possible_agents
            ):
                break

            actions = {}
            # for agent_id, agent_obs in obs.items():

            #     action = algorithm.compute_single_action(agent_obs, policy_id="shared_policy")
            #     actions[agent_id] = action
            for agent_id, agent_obs in obs.items():
                # [수정 포인트] 학습 설정과 동일하게 Policy ID를 결정해야 함
                # config에서 설정한 로직: "0"이 포함되면 policy_left, 아니면 policy_right
                target_policy_id = "policy_left" if "0" in agent_id else "policy_right"
                
                action = algorithm.compute_single_action(
                    agent_obs, 
                    policy_id=target_policy_id, # 수정됨
                    explore=False
                )
                actions[agent_id] = action
                
            obs, rewards, terminations, truncations, infos = env.step(actions)

            if (step_i % every_n_steps) == 0:
                if len(frames) >= max_frames:
                    break
                fr = env.render()
                if fr is not None:
                    frames.append(fr)

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
    """
    evaluation 5번마다 GIF 1개 저장 

    사용 예)
      .callbacks(lambda: EvalEveryNGifCallbacks(
          out_dir=".../results/gifs",
          every_n_evals=5,
          max_cycles=500,
      ))
    """

    def __init__(
        self,
        out_dir: str,
        every_n_evals: int = 5,
        max_cycles: int = 500,
        every_n_steps: int = 2,
        max_frames: int = 400,
        fps: int = 30,
        filename_prefix: str = "eval5",
    ):
        super().__init__()
        self.out_dir = out_dir
        self.every_n_evals = every_n_evals
        self.max_cycles = max_cycles
        self.every_n_steps = every_n_steps
        self.max_frames = max_frames
        self.fps = fps
        self.filename_prefix = filename_prefix

        self.eval_count = 0
        os.makedirs(self.out_dir, exist_ok=True)

    def on_train_result(self, *, algorithm, result, **kwargs):
        """
        RLlib에서 evaluation이 돌면 보통 result 안에 "evaluation" 키가 포함됨.
        (evaluation_interval 설정되어 있을 때)
        """
        training_iter = int(result.get("training_iteration", 0))

        if "evaluation" not in result:
            return

        self.eval_count += 1
        if (self.eval_count % self.every_n_evals) != 0:
            return

        out_path = os.path.join(
            self.out_dir,
            f"{self.filename_prefix}_{self.eval_count:04d}_iter{training_iter:06d}.gif",
        )

        rollout_and_save_gif(
            algorithm=algorithm,
            out_path=out_path,
            max_cycles=self.max_cycles,
            every_n_steps=self.every_n_steps,
            max_frames=self.max_frames,
            fps=self.fps,
        )