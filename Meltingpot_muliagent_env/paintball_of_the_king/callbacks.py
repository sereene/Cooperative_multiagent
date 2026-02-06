import os
import gc
import copy
import imageio.v2 as imageio
import numpy as np
import torch
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from env_utils import env_creator

# [추가] WandB 연동을 위해 임포트
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
        
        # 가중치 저장소
        self.history_dir = os.path.join(out_dir, "policy_history")
        os.makedirs(self.history_dir, exist_ok=True)
        self.history_index = [] # (iteration, path) 튜플 저장

    def _get_pure_weights(self, algorithm, policy_id):
        """
        [안전장치] RLLib 버전에 따라 get_weights가 {'policy_id': weights} 형태일 수도 있고
        그냥 weights일 수도 있습니다. 이를 확실하게 벗겨내어 반환하는 함수입니다.
        """
        weights = algorithm.get_weights(policy_id)
        # 만약 반환된 weights가 dict이고, 그 안에 policy_id 키가 있다면 그 안의 값을 씁니다.
        if isinstance(weights, dict) and policy_id in weights:
            return weights[policy_id]
        return weights

    def on_train_result(self, *, algorithm, result, **kwargs):
        iteration = result.get("training_iteration", 0)
        
        # ---------------------------------------------------------------------
        # 1. [History] 현재 정책 저장 (매 update_interval_iter 마다)
        # ---------------------------------------------------------------------
        if iteration % self.update_interval_iter == 0:
            # [안전장치] 가중치 추출
            main_weights = self._get_pure_weights(algorithm, "main_policy")
            
            save_path = os.path.join(self.history_dir, f"weights_iter_{iteration}.pt")
            
            # CPU로 이동하여 저장 (Tensor인 경우)
            cpu_weights = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in main_weights.items()}
            torch.save(cpu_weights, save_path)
            
            self.history_index.append((iteration, save_path))
            # 20개 유지
            if len(self.history_index) > 20:
                old_iter, old_path = self.history_index.pop(0)
                if os.path.exists(old_path):
                    os.remove(old_path)

        # ---------------------------------------------------------------------
        # 2. [Self-Play] Opponent 업데이트 (현재 Main 정책 복사)
        # ---------------------------------------------------------------------
        if iteration > 0 and iteration % self.update_interval_iter == 0:
            print(f"\n🔄 [Self-Play] Updating Opponent to match Main Policy (Iter {iteration})")
            
            # [안전장치] 가중치 추출
            main_weights = self._get_pure_weights(algorithm, "main_policy")
            
            # set_weights는 {"policy_id": pure_weights} 형태를 받습니다.
            algorithm.set_weights({"opponent_policy": main_weights})

        # ---------------------------------------------------------------------
        # 3. [Past Evaluation] 과거의 나랑 싸우기 (100 iter 전 모델)
        # ---------------------------------------------------------------------
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
                
                # (1) 현재 Opponent 백업 (안전하게 추출)
                original_opponent_weights = copy.deepcopy(
                    self._get_pure_weights(algorithm, "opponent_policy")
                )
                
                # (2) 과거의 나 로드 [수정된 부분: weights_only=False 추가]
                try:
                    past_weights = torch.load(best_ckpt, weights_only=False)
                except TypeError:
                    # 구버전 PyTorch 호환성을 위한 예외 처리
                    past_weights = torch.load(best_ckpt)
                
                # [안전장치] 혹시 과거에 잘못 저장된 파일(policy 키가 포함된 파일)일 경우 구조 맞춤
                if "main_policy" in past_weights:
                    past_weights = past_weights["main_policy"]
                
                # Opponent에 주입
                algorithm.set_weights({"opponent_policy": past_weights})
                
                # (3) 승부
                win_rate, avg_score = self._run_duel(algorithm, num_matches=3)
                
                # (4) 결과 기록
                result["custom_metrics"]["win_rate_vs_past_100"] = win_rate
                result["custom_metrics"]["score_vs_past_100"] = avg_score
                print(f"   >>> Result: Win Rate {win_rate*100:.1f}%, Score {avg_score:.1f}")

                # (5) Opponent 원상복구
                algorithm.set_weights({"opponent_policy": original_opponent_weights})

        # ---------------------------------------------------------------------
        # 4. GIF 생성 및 WandB 업로드
        # ---------------------------------------------------------------------
        if "evaluation" in result:
            self.eval_count += 1
            if self.eval_count % 5 == 0:
                out_path = os.path.join(self.out_dir, f"eval_{self.eval_count:04d}_iter{iteration:06d}.gif")
                rollout_and_save_gif(algorithm=algorithm, out_path=out_path, max_cycles=self.max_cycles)

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _run_duel(self, algorithm, num_matches=1):
        """평가를 위해 별도로 게임을 돌리는 함수"""
        env = env_creator({"substrate": "paintball__king_of_the_hill"})
        red_wins = 0
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
                            agent_obs, state=agent_states[agent_id], policy_id=policy_id, explore=False
                        )
                        actions[agent_id] = res[0] if isinstance(res, tuple) else res
                        agent_states[agent_id] = res[1] if isinstance(res, tuple) else agent_states[agent_id]
                    
                    obs, rewards, terms, truncs, _ = env.step(actions)
                    
                    for (aid, _), r in rewards.items():
                        if aid in ["player_0", "player_2"]: score_red += r
                        else: score_blue += r

                    if any(terms.values()) or all(truncs.values()) or not obs:
                        break
                
                if score_red > score_blue: red_wins += 1
                total_score_diff += (score_red - score_blue)
                
        finally:
            env.close()
            del env
        
        return red_wins / num_matches, total_score_diff / num_matches


def rollout_and_save_gif(algorithm, out_path, max_cycles=1000, fps=30):
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
        print(f"[GIF] Saved: {out_path}")
        
        try:
            if wandb.run is not None:
                wandb.log({
                    "evaluation/gameplay_gif": wandb.Video(out_path, fps=fps, format="gif", caption="Latest Evaluation Replay")
                })
                print(f"[WandB] GIF uploaded successfully.")
        except Exception as e:
            print(f"[Warning] Failed to upload GIF to WandB: {e}")

    finally:
        env.close()
        del env, agent_states, obs, frames
        gc.collect()
