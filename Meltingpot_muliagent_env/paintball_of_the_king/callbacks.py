import os
import gc
import imageio.v2 as imageio
import numpy as np
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from env_utils import env_creator

class MeltingPotCallbacks(DefaultCallbacks):
    def on_episode_start(self, *, worker, base_env, policies, episode, **kwargs):
        # 1. 사망 횟수 초기화
        for i in range(4):
            episode.user_data[f"deaths_player_{i}"] = 0
            
        # 2. 점령 스텝 카운터 초기화
        episode.user_data["red_occupation_steps"] = 0

    def on_episode_step(self, *, worker, base_env, policies, episode, **kwargs):
        # 1. 사망 횟수 집계
        for i in range(4):
            agent_id = f"player_{i}"
            info = episode.last_info_for(agent_id)
            if info and info.get("was_zapped", False):
                episode.user_data[f"deaths_{agent_id}"] += 1

        # 2. 점령 여부 확인
        r0 = 0.0
        try:
            r0 = episode.prev_reward_for("player_0")
        except AttributeError:
            for (aid, pid), reward in episode.agent_rewards.items():
                if aid == "player_0":
                    r0 = reward
                    break
        
        if r0 > 0:
            episode.user_data["red_occupation_steps"] += 1

    def on_episode_end(self, *, worker, base_env, policies, episode, **kwargs):
        # 1. 총점 기록
        episode.custom_metrics["score"] = episode.total_reward
        
        # 2. 개별 보상 기록
        for (agent_id, policy_id), reward in episode.agent_rewards.items():
            episode.custom_metrics[f"reward_{agent_id}"] = reward
        
        # 3. 사망 횟수(Deaths) 기록 -> 이것의 총합이 곧 교전 횟수(Zaps)입니다.
        total_deaths = 0
        for i in range(4):
            key = f"deaths_player_{i}"
            death_count = episode.user_data.get(key, 0)
            episode.custom_metrics[key] = death_count
            total_deaths += death_count

        # [추가] 전체 교전 활성화 정도 (높을수록 서로 잘 싸우는 것)
        episode.custom_metrics["total_zaps_in_episode"] = total_deaths

        # 4. 점령률 계산
        occupation_steps = episode.user_data.get("red_occupation_steps", 0)
        episode_len = episode.length if episode.length > 0 else 1000
        
        occupation_rate = occupation_steps / episode_len if episode_len > 0 else 0
        episode.custom_metrics["occupation_rate"] = occupation_rate

def rollout_and_save_gif(
    *,
    algorithm,
    out_path: str,
    max_cycles: int = 1000,
    every_n_steps: int = 4,
    max_frames: int = 1000,
    fps: int = 30,
):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    # [주의] env_utils에서 수정된 env_creator를 가져와야 함
    env = env_creator({"substrate": "paintball__king_of_the_hill"})
    frames = []
    agent_states = {}

    try:
        obs, infos = env.reset()
        step_i = 0
        fr0 = env.par_env.render() 
        if fr0 is not None: frames.append(fr0)
        
        terminations = {a: False for a in env.par_env.possible_agents}
        truncations = {a: False for a in env.par_env.possible_agents}

        while True:
            if not obs: break
            actions = {}
            for agent_id, agent_obs in obs.items():
                policy_id = algorithm.config.policy_mapping_fn(agent_id)
                
                if agent_id not in agent_states:
                    policy = algorithm.get_policy(policy_id)
                    agent_states[agent_id] = policy.get_initial_state()

                # [수정된 부분] 결과가 튜플인지 값 하나인지 확인하여 처리
                result = algorithm.compute_single_action(
                    agent_obs, 
                    state=agent_states[agent_id],
                    policy_id=policy_id, 
                    explore=True
                )
                
                # 결과 타입에 따른 분기 처리
                if isinstance(result, tuple) and len(result) >= 2:
                    # (action, state, info) 튜플인 경우 (Recurrent Policy)
                    action = result[0]
                    state_out = result[1]
                else:
                    # Action 값 하나만 온 경우 (Stateless Policy)
                    action = result
                    state_out = agent_states[agent_id] # 상태 변화 없음 (빈 리스트 유지)

                actions[agent_id] = action
                agent_states[agent_id] = state_out

            obs, rewards, terminations, truncations, infos = env.step(actions)

            if (step_i % every_n_steps) == 0:
                if len(frames) >= max_frames: break
                fr = env.par_env.render()
                if fr is not None: frames.append(fr)
            step_i += 1
            if any(terminations.values()) or all(truncations.values()) or len(obs) == 0:
                break
            if step_i >= max_cycles:
                break

        if frames:
            imageio.mimsave(out_path, frames, fps=fps)
            print(f"[GIF] Saved: {out_path} (Frames: {len(frames)})")
    finally:
        try:
            env.close()
            gc.collect()
        except Exception:
            pass

class GifCallbacks(MeltingPotCallbacks):
    def __init__(self, out_dir: str, every_n_evals: int = 5, max_cycles: int = 1000):
        super().__init__()
        self.out_dir = out_dir
        self.every_n_evals = every_n_evals
        self.max_cycles = max_cycles
        self.eval_count = 0
        os.makedirs(self.out_dir, exist_ok=True)

    def on_train_result(self, *, algorithm, result, **kwargs):
        if "evaluation" not in result: return
        
        training_iter = int(result.get("training_iteration", 0))
        self.eval_count += 1
        
        if (self.eval_count % self.every_n_evals) == 0:
            out_path = os.path.join(self.out_dir, f"eval_{self.eval_count:04d}_iter{training_iter:06d}.gif")
            print(f"Generating GIF at {out_path}...")
            rollout_and_save_gif(
                algorithm=algorithm, 
                out_path=out_path, 
                max_cycles=self.max_cycles
            )