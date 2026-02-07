import os
import gc
import copy
import imageio.v2 as imageio
import numpy as np
import torch
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from env_utils import env_creator
import wandb 

class MeltingPotCallbacks(DefaultCallbacks):
    """기본 메트릭 집계 콜백"""
    def on_episode_start(self, *, worker, base_env, policies, episode, **kwargs):
        for i in range(4):
            episode.user_data[f"deaths_player_{i}"] = 0
        episode.user_data["red_occupation_steps"] = 0
        episode.user_data["blue_occupation_steps"] = 0

    def on_episode_step(self, *, worker, base_env, policies, episode, **kwargs):
        # 1. 교전 횟수 집계
        for i in range(4):
            agent_id = f"player_{i}"
            info = episode.last_info_for(agent_id)
            if info and "events" in info:
                for event in info["events"]:
                    if isinstance(event, dict) and event.get("name") == "removal":
                        episode.user_data[f"deaths_{agent_id}"] += 1

        # 2. 점령률 집계
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
# [핵심] Self-Play + Past Evaluation + WandB GIF 업로드
# -----------------------------------------------------------------------------
class SelfPlayCallback(MeltingPotCallbacks):
    def __init__(self, out_dir: str, update_interval_iter: int = 50, max_cycles: int = 1000):
        super().__init__()
        self.out_dir = out_dir
        self.update_interval_iter = update_interval_iter
        self.max_cycles = max_cycles
        self.eval_count = 0
        
        # [수정] WandB init 제거 (RolloutWorker 충돌 방지)
        # self.history_dir 등 로컬 경로 설정만 수행
        self.history_dir = os.path.join(out_dir, "policy_history")
        os.makedirs(self.history_dir, exist_ok=True)
        self.history_index = [] 

    def _get_pure_weights(self, algorithm, policy_id):
        weights = algorithm.get_weights(policy_id)
        if isinstance(weights, dict) and policy_id in weights:
            return weights[policy_id]
        return weights

    def on_train_result(self, *, algorithm, result, **kwargs):
        # [중요] on_train_result는 오직 Driver(메인 프로세스)에서만 실행됨.
        # 따라서 여기서 wandb.log를 호출하는 것은 안전함.
        
        iteration = result.get("training_iteration", 0)
        current_step = result.get("timesteps_total", 0)
        
        # 0. 기본 학습 지표 로깅
        if wandb.run is not None:
            log_payload = {
                "train/episode_reward_mean": result.get("episode_reward_mean"),
                "train/episode_len_mean": result.get("episode_len_mean"),
                "train/training_iteration": iteration,
                "train/timesteps_total": current_step
            }
            if "custom_metrics" in result:
                for k, v in result["custom_metrics"].items():
                    log_payload[f"custom_metrics/{k}"] = v
            
            try:
                wandb.log(log_payload, step=current_step)
            except Exception as e:
                print(f"[Warning] WandB log failed: {e}")

        # 1. [History] 정책 저장
        if iteration % self.update_interval_iter == 0:
            main_weights = self._get_pure_weights(algorithm, "main_policy")
            save_path = os.path.join(self.history_dir, f"weights_iter_{iteration}.pt")
            
            cpu_weights = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in main_weights.items()}
            torch.save(cpu_weights, save_path)
            
            self.history_index.append((iteration, save_path))
            if len(self.history_index) > 20:
                old_iter, old_path = self.history_index.pop(0)
                if os.path.exists(old_path):
                    os.remove(old_path)

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

                # WandB 로깅
                if wandb.run is not None:
                    wandb.log({
                        "eval_vs_past/win_rate": win_rate,
                        "eval_vs_past/score_diff": avg_score,
                    }, step=current_step)

                algorithm.set_weights({"opponent_policy": original_opponent_weights})

        # 4. [GIF 생성]
        if iteration > 0 and iteration % self.update_interval_iter == 0:
            self.eval_count += 1
            if self.eval_count % 5 == 0:
                out_path = os.path.join(self.out_dir, f"eval_{self.eval_count:04d}_iter{iteration:06d}.gif")
                print(f"🎬 Generating GIF to {out_path}...")
                # GIF 생성 및 업로드
                rollout_and_save_gif(algorithm=algorithm, out_path=out_path, max_cycles=self.max_cycles, step=current_step)

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
                        
                        res = algorithm.compute_single_action(
                            agent_obs, state=agent_states[agent_id], policy_id=policy_id, explore=True
                        )
                        actions[agent_id] = res[0] if isinstance(res, tuple) else res
                        agent_states[agent_id] = res[1] if isinstance(res, tuple) else agent_states[agent_id]
                    
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

def rollout_and_save_gif(algorithm, out_path, max_cycles=1000, fps=30, step=None):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    env = env_creator({"substrate": "paintball__king_of_the_hill"})
    frames = []
    agent_states = {}
    
    try:
        obs, _ = env.reset()
        if not obs: return
        
        fr = env.par_env.render()
        if fr is not None: frames.append(fr)
        
        for _ in range(max_cycles):
            actions = {}
            for agent_id, agent_obs in obs.items():
                policy_id = algorithm.config.policy_mapping_fn(agent_id)
                if agent_id not in agent_states:
                    policy = algorithm.get_policy(policy_id)
                    agent_states[agent_id] = policy.get_initial_state()
                
                res = algorithm.compute_single_action(
                    agent_obs, state=agent_states[agent_id], policy_id=policy_id, explore=True
                )
                actions[agent_id] = res[0] if isinstance(res, tuple) else res
                agent_states[agent_id] = res[1] if isinstance(res, tuple) else agent_states[agent_id]
            
            obs, _, terms, truncs, _ = env.step(actions)
            
            fr = env.par_env.render()
            if fr is not None: frames.append(fr)
            
            if any(terms.values()) or all(truncs.values()) or not obs:
                break
        
        imageio.mimsave(out_path, frames, fps=fps)
        print(f"[GIF] Saved to disk: {out_path}")
        
        # [수정] WandB run 상태 확인 후 로깅
        if wandb.run is not None:
            try:
                video = wandb.Video(out_path, fps=fps, format="gif", caption=f"Step {step}")
                if step is not None:
                    wandb.log({"evaluation/gameplay_gif": video}, step=step)
                else:
                    wandb.log({"evaluation/gameplay_gif": video})
                print(f"[WandB] 🟢 Uploaded GIF at step {step}")
            except Exception as e:
                print(f"[WandB] 🔴 Failed to upload GIF: {e}")

    finally:
        env.close()
        del env, agent_states, obs, frames
        gc.collect()
