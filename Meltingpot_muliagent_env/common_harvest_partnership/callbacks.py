import os
import gc
import imageio.v2 as imageio
import numpy as np
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from env_utils import env_creator

class MeltingPotCallbacks(DefaultCallbacks):
    def on_episode_start(self, *, worker, base_env, policies, episode, **kwargs):
        # 사망 횟수 초기화
        episode.user_data["deaths_player_0"] = 0
        episode.user_data["deaths_player_1"] = 0

    def on_episode_step(self, *, worker, base_env, policies, episode, **kwargs):
        # 매 스텝마다 Wrapper가 보낸 info 확인
        for agent_id in ["player_0", "player_1"]:
            # 해당 에이전트의 마지막 info 가져오기
            info = episode.last_info_for(agent_id)
            
            # Wrapper에서 was_zapped=True로 설정했다면 카운트 증가
            if info and info.get("was_zapped", False):
                episode.user_data[f"deaths_{agent_id}"] += 1

    def on_episode_end(self, *, worker, base_env, policies, episode, **kwargs):
        # 1. 총점 기록
        episode.custom_metrics["score"] = episode.total_reward
        
        # 2. 개별 보상 기록 (튜플 키 문제 해결)
        # episode.agent_rewards는 {(agent_id, policy_id): reward} 형태임
        for (agent_id, policy_id), reward in episode.agent_rewards.items():
            episode.custom_metrics[f"reward_{agent_id}"] = reward

        # 3. 사망 횟수 기록 (WandB에 deaths_player_0_mean 등으로 표시됨)
        episode.custom_metrics["deaths_player_0"] = episode.user_data.get("deaths_player_0", 0)
        episode.custom_metrics["deaths_player_1"] = episode.user_data.get("deaths_player_1", 0)

def rollout_and_save_gif(
    *,
    algorithm,
    out_path: str,
    max_cycles: int = 500,
    every_n_steps: int = 4,
    max_frames: int = 300,
    fps: int = 30,
):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    env = env_creator()
    frames = []
    
    # [수정 1] 각 에이전트의 LSTM 상태(h, c)를 저장할 딕셔너리 생성
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
                
                # [수정 2] 해당 에이전트의 상태가 없으면 초기화 (Initial State)
                if agent_id not in agent_states:
                    policy = algorithm.get_policy(policy_id)
                    # 모델에서 정의한 get_initial_state() 호출 -> [zeros, zeros] 반환
                    agent_states[agent_id] = policy.get_initial_state()

                # [수정 3] state를 입력으로 넣고, 다음 스텝을 위한 new_state를 반환받음
                # compute_single_action 반환값: (action, state_out, extra_info)
                action, state_out, _ = algorithm.compute_single_action(
                    agent_obs, 
                    state=agent_states[agent_id],  # 현재 상태 전달
                    policy_id=policy_id, 
                    explore=False
                )
                
                actions[agent_id] = action
                agent_states[agent_id] = state_out # 다음 스텝을 위해 상태 업데이트

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
            print(f"[GIF] Saved: {out_path}")
    finally:
        try:
            env.close()
            gc.collect()
        except Exception:
            pass

class GifCallbacks(MeltingPotCallbacks):
    def __init__(self, out_dir: str, every_n_evals: int = 5, max_cycles: int = 500):
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