import os
import gc
import copy
import numpy as np
import torch
import imageio.v2 as imageio
import wandb 
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from env_utils import env_creator

class MeltingPotCallbacks(DefaultCallbacks):
    """기본 메트릭 집계 콜백"""
    def on_episode_start(self, *, worker, base_env, policies, episode, **kwargs):
        for i in range(4):
            episode.user_data[f"deaths_player_{i}"] = 0
        episode.user_data["red_occupation_steps"] = 0
        episode.user_data["blue_occupation_steps"] = 0

    def on_episode_step(self, *, worker, base_env, policies, episode, **kwargs):
        for i in range(4):
            agent_id = f"player_{i}"
            info = episode.last_info_for(agent_id)
            if info and "events" in info:
                for event in info["events"]:
                    if isinstance(event, dict) and event.get("name") == "removal":
                        episode.user_data[f"deaths_{agent_id}"] += 1

        r0 = 0.0
        try:
            r0 = episode.prev_reward_for("player_0")
        except AttributeError:
            for (aid, _), reward in episode.agent_rewards.items():
                if aid == "player_0":
                    r0 = reward
                    break
        
        if r0 > 0:
            episode.user_data["red_occupation_steps"] += 1
        elif r0 < 0:
            episode.user_data["blue_occupation_steps"] += 1

    def on_episode_end(self, *, worker, base_env, policies, episode, **kwargs):
        episode.custom_metrics["score"] = episode.total_reward
        total_deaths = sum(episode.user_data.get(f"deaths_player_{i}", 0) for i in range(4))
        episode.custom_metrics["total_zaps_in_episode"] = total_deaths
        
        ep_len = max(episode.length, 1)
        episode.custom_metrics["occupation_rate_red"] = episode.user_data["red_occupation_steps"] / ep_len
        episode.custom_metrics["occupation_rate_blue"] = episode.user_data["blue_occupation_steps"] / ep_len

# -----------------------------------------------------------------------------
# [수정됨] GIF 생성 함수 (안전한 Unpacking 적용)
# -----------------------------------------------------------------------------
def rollout_and_save_gif(
    *,
    algorithm,
    out_path: str,
    max_cycles: int = 1000,
    every_n_steps: int = 4,   
    max_frames: int = 300,    
    fps: int = 30,
    upload_to_wandb: bool = True,
    wandb_key: str = "evaluation/gif",
    step: int = 0
):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    # Paintball 환경 생성
    env = env_creator({"substrate": "paintball__king_of_the_hill"})
    frames = []
    
    # LSTM 상태 관리
    agent_states = {}

    try:
        obs, infos = env.reset()
        step_i = 0
        
        # 첫 프레임 렌더링
        try:
            fr0 = env.par_env.render() 
            if fr0 is not None: frames.append(fr0)
        except Exception:
            pass
        
        terminations = {a: False for a in env.par_env.possible_agents}
        truncations = {a: False for a in env.par_env.possible_agents}

        while True:
            if not obs: break
            actions = {}
            for agent_id, agent_obs in obs.items():
                policy_id = algorithm.config.policy_mapping_fn(agent_id)
                
                # 해당 에이전트의 상태가 없으면 초기화
                if agent_id not in agent_states:
                    policy = algorithm.get_policy(policy_id)
                    agent_states[agent_id] = policy.get_initial_state()

                # [핵심 수정] full_fetch=True를 사용하여 항상 튜플 반환을 강제함
                result = algorithm.compute_single_action(
                    agent_obs, 
                    state=agent_states[agent_id], 
                    policy_id=policy_id, 
                    explore=False,
                    full_fetch=True 
                )
                
                # [안전 장치] 결과값 타입 체크 및 언패킹
                if isinstance(result, tuple) and len(result) >= 3:
                    action, state_out, _ = result
                else:
                    # 만약 튜플이 아닌 경우 (스칼라 액션만 반환된 경우 등)
                    action = result
                    state_out = agent_states[agent_id] # 상태 유지

                actions[agent_id] = action
                agent_states[agent_id] = state_out 

            obs, rewards, terminations, truncations, infos = env.step(actions)

            # 프레임 수집
            if (step_i % every_n_steps) == 0:
                if len(frames) >= max_frames: break
                try:
                    fr = env.par_env.render()
                    if fr is not None: frames.append(fr)
                except Exception:
                    pass
            
            step_i += 1
            if any(terminations.values()) or all(truncations.values()) or len(obs) == 0:
                break
            if step_i >= max_cycles:
                break

        if frames:
            # GIF 저장
            imageio.mimsave(out_path, frames, fps=fps, loop=0)
            print(f"[GIF] Saved: {out_path}")
            
            # WandB 업로드
            if upload_to_wandb and wandb.run is not None:
                try:
                    wandb.log({
                        wandb_key: wandb.Video(out_path, fps=fps, format="gif", caption=f"Step {step}")
                    }, step=step)
                    print(f"[WandB] Uploaded GIF to {wandb_key}")
                except Exception as e:
                    print(f"[Warning] WandB upload failed: {e}")

    finally:
        try:
            env.close()
            gc.collect()
        except Exception:
            pass

# -----------------------------------------------------------------------------
# Self-Play 및 GIF Logging Callback
# -----------------------------------------------------------------------------
class SelfPlayCallback(MeltingPotCallbacks):
    def __init__(self, out_dir: str, update_interval_iter: int = 50, max_cycles: int = 1000):
        super().__init__()
        self.out_dir = out_dir 
        self.update_interval_iter = update_interval_iter
        self.max_cycles = max_cycles
        self.eval_count = 0
        
        self.history_dir = os.path.join(self.out_dir, "policy_history")
        os.makedirs(self.history_dir, exist_ok=True)
        self.history_index = [] 

    def _get_pure_weights(self, algorithm, policy_id):
        weights = algorithm.get_weights(policy_id)
        if isinstance(weights, dict) and policy_id in weights:
            return weights[policy_id]
        return weights

    def on_train_result(self, *, algorithm, result, **kwargs):
        iteration = result.get("training_iteration", 0)
        current_step = result.get("timesteps_total", 0)
        
        # 1. [History] 정책 저장
        if iteration % self.update_interval_iter == 0:
            main_weights = self._get_pure_weights(algorithm, "main_policy")
            save_path = os.path.join(self.history_dir, f"weights_iter_{iteration}.pt")
            
            # CPU로 이동하여 저장
            cpu_weights = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in main_weights.items()}
            torch.save(cpu_weights, save_path)
            
            self.history_index.append((iteration, save_path))
            if len(self.history_index) > 20:
                old_iter, old_path = self.history_index.pop(0)
                if os.path.exists(old_path):
                    try:
                        os.remove(old_path)
                    except OSError:
                        pass

        # 2. [Self-Play] Opponent 업데이트
        if iteration > 0 and iteration % self.update_interval_iter == 0:
            print(f"\n🔄 [Self-Play] Updating Opponent to match Main Policy (Iter {iteration})")
            main_weights = self._get_pure_weights(algorithm, "main_policy")
            algorithm.set_weights({"opponent_policy": main_weights})

        # 3. [Past Evaluation] 과거 정책 대결
        target_lag = 100 
        if iteration >= target_lag and iteration % self.update_interval_iter == 0:
            target_iter = iteration - target_lag
            
            best_ckpt = None
            for it, path in self.history_index:
                if abs(it - target_iter) < 25:
                    best_ckpt = path
                    break
            
            if best_ckpt and os.path.exists(best_ckpt):
                print(f"⚔️ [Past-Eval] Fighting against checkpoint from Iter {target_iter}...")
                
                original_opponent_weights = copy.deepcopy(
                    self._get_pure_weights(algorithm, "opponent_policy")
                )
                
                try:
                    past_weights = torch.load(best_ckpt, weights_only=False)
                except TypeError:
                    past_weights = torch.load(best_ckpt)
                
                if "main_policy" in past_weights:
                    past_weights = past_weights["main_policy"]
                
                algorithm.set_weights({"opponent_policy": past_weights})
                
                win_rate, avg_score = self._run_duel(algorithm, num_matches=3)
                print(f"   >>> Result: Win Rate {win_rate*100:.1f}%, Score Diff {avg_score:.1f}")

                if "custom_metrics" not in result:
                    result["custom_metrics"] = {}
                result["custom_metrics"]["eval_vs_past/win_rate"] = win_rate
                result["custom_metrics"]["eval_vs_past/score_diff"] = avg_score

                algorithm.set_weights({"opponent_policy": original_opponent_weights})

        # 4. [GIF 생성]
        if iteration > 0 and iteration % self.update_interval_iter == 0:
            self.eval_count += 1
            if self.eval_count % 5 == 0:
                out_path = os.path.join(self.out_dir, f"eval_{self.eval_count:04d}_iter{iteration:06d}.gif")
                print(f"🎬 Generating GIF at {out_path}...")
                
                rollout_and_save_gif(
                    algorithm=algorithm, 
                    out_path=out_path, 
                    max_cycles=self.max_cycles,
                    step=current_step,
                    wandb_key="test_rollout/video"
                )

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _run_duel(self, algorithm, num_matches=1):
        env = env_creator({"substrate": "paintball__king_of_the_hill"})
        red_wins = 0.0
        total_score_diff = 0
        
        try:
            for _ in range(num_matches):
                obs, _ = env.reset()
                agent_states = {}
                score_red = 0
                score_blue = 0
                
                for _ in range(self.max_cycles):
                    actions = {}
                    for agent_id, agent_obs in obs.items():
                        policy_id = algorithm.config.policy_mapping_fn(agent_id)
                        if agent_id not in agent_states:
                            policy = algorithm.get_policy(policy_id)
                            agent_states[agent_id] = policy.get_initial_state()
                        
                        # [핵심 수정] 여기도 full_fetch=True 및 안전한 언패킹 적용
                        result = algorithm.compute_single_action(
                            agent_obs, 
                            state=agent_states[agent_id], 
                            policy_id=policy_id, 
                            explore=True, # 대결 모드이므로 explore=True
                            full_fetch=True 
                        )
                        
                        if isinstance(result, tuple) and len(result) >= 3:
                            action, state_out, _ = result
                        else:
                            action = result
                            state_out = agent_states[agent_id]

                        actions[agent_id] = action
                        agent_states[agent_id] = state_out
                    
                    obs, rewards, terms, truncs, _ = env.step(actions)
                    
                    for aid, r in rewards.items():
                        if aid in ["player_0", "player_2"]: score_red += r
                        else: score_blue += r

                    if any(terms.values()) or all(truncs.values()) or not obs:
                        break
                
                if score_red > score_blue:
                    red_wins += 1.0
                elif score_red == score_blue and (score_red > 0 or score_blue > 0):
                    red_wins += 0.5
                elif score_red == score_blue:
                    red_wins += 0.5

                total_score_diff += (score_red - score_blue)
                
        finally:
            env.close()
            del env
        
        return red_wins / num_matches, total_score_diff / num_matches