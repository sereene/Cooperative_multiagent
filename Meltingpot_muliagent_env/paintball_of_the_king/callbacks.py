import os
import gc
import copy
import numpy as np
import torch
import imageio.v2 as imageio
import tempfile  # tempfile 모듈 추가
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
# [핵심] Self-Play 및 Video Logging Callback
# -----------------------------------------------------------------------------
class SelfPlayCallback(MeltingPotCallbacks):
    def __init__(self, out_dir: str, update_interval_iter: int = 50, max_cycles: int = 1000):
        super().__init__()
        self.out_dir = out_dir 
        self.update_interval_iter = update_interval_iter
        self.max_cycles = max_cycles
        self.eval_count = 0
        
        # [설정] WandB 로깅 시 사용할 접두사 (gif_name 역할)
        self.video_name = "test_rollout" 
        
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

                if "custom_metrics" not in result:
                    result["custom_metrics"] = {}
                result["custom_metrics"]["eval_vs_past/win_rate"] = win_rate
                result["custom_metrics"]["eval_vs_past/score_diff"] = avg_score

                algorithm.set_weights({"opponent_policy": original_opponent_weights})

        # 4. [Video 생성] - tempfile 버전 로직 적용
        if iteration > 0 and iteration % self.update_interval_iter == 0:
            self.eval_count += 1
            if self.eval_count % 5 == 0:
                print(f"🎬 Generating Video (Iter {iteration})...")
                
                # rollout_and_save_video 함수 호출
                # target_dir 인자로 self.out_dir 전달
                rollout_and_save_video(
                    algorithm=algorithm, 
                    target_dir=self.out_dir, 
                    max_cycles=self.max_cycles, 
                    step=current_step, 
                    epoch=iteration,
                    video_name=self.video_name 
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

# -----------------------------------------------------------------------------
# [함수] Video Rollout 및 WandB 업로드 (tempfile 버전)
# -----------------------------------------------------------------------------
def rollout_and_save_video(algorithm, target_dir, max_cycles=1000, fps=30, step=None, epoch=None, video_name="test_rollout"):
    if wandb is None or wandb.run is None:
        return

    # Create Environment
    env = env_creator({"substrate": "paintball__king_of_the_hill"})
    
    frames = []
    total_reward = 0.0
    t = 0
    agent_states = {}

    try:
        # 1. Start / Reset
        try:
            obs, info = env.reset()
        except Exception as e:
            print(f"[Video] start_failed: {repr(e)}")
            return
        
        if not obs:
            return

        done = False
        
        # 2. Loop
        while (not done) and (t < max_cycles):
            # Render offscreen
            try:
                frame = env.par_env.render()
                if frame is not None:
                    frames.append(frame)
            except Exception:
                pass

            # Compute Actions
            actions = {}
            try:
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
            except Exception as e:
                print(f"[Video] action_failed: {repr(e)}")
                break

            if not actions:
                print("[Video] action_failed: actions is empty")
                break

            # Step
            try:
                obs, rewards, terms, truncs, info = env.step(actions)
            except Exception as e:
                print(f"[Video] step_failed: {repr(e)}")
                break
            
            # Aggregate Reward
            try:
                if rewards:
                    total_reward += sum(rewards.values())
            except Exception:
                pass

            # Check Done
            try:
                if any(terms.values()) or all(truncs.values()) or not obs:
                    done = True
            except Exception:
                done = False
            
            t += 1

    finally:
        env.close()
        del env

    if len(frames) < 2:
        print(f"[Video] Rollout too short: frames={len(frames)}")
        return

    # 3. Write MP4 using tempfile (as requested)
    # 디렉토리가 없으면 생성
    try:
        os.makedirs(target_dir, exist_ok=True)
        log_dir = target_dir
    except Exception:
        log_dir = "."

    # tempfile 생성 (delete=False로 설정하여 파일 유지)
    with tempfile.NamedTemporaryFile(suffix=".mp4", dir=log_dir, delete=False) as f:
        video_path = f.name

    # Save using imageio (mimsave)
    try:
        # macro_block_size=None ensures better compatibility for odd dimensions
        imageio.mimsave(video_path, frames, fps=fps, macro_block_size=None)
        print(f"[Video] Saved to local disk (temp): {video_path}")
    except Exception as e:
        try:
            if os.path.exists(video_path):
                os.remove(video_path)
        except Exception:
            pass
        print(f"[Video] write_failed(imageio): {repr(e)}")
        return

    # 4. Upload to WandB
    # 요청하신 코드의 dict 키 형식 그대로 적용
    try:
        wandb.log(
            {
                f"{video_name}/video": wandb.Video(video_path, fps=fps, format="mp4", caption=f"Epoch {epoch}"),
                f"{video_name}/total_reward": float(total_reward),
                f"{video_name}/length": int(t),
                f"{video_name}/epoch": int(epoch),
            },
            step=int(step) if step is not None else None,
        )
        print(f"[WandB] 🟢 Uploaded Video to '{video_name}/video' at step {step}")
    except Exception as e:
        print(f"[Video] wandb_log_failed: {repr(e)}")

    # Cleanup memory
    del frames
    gc.collect()
