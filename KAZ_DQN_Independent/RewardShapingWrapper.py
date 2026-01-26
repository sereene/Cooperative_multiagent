import math
import pygame
import gymnasium as gym
import numpy as np
from pettingzoo.utils.wrappers import BaseParallelWrapper

class RewardShapingWrapper(BaseParallelWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.prev_min_dists = {}
        self.prev_zombie_count = 0
        
    def _clip_observation(self, obs):
        """
        Recursively clips observations to be strictly within [-1, 1].
        Handles dictionary observations (multi-agent) recursively.
        """
        if isinstance(obs, dict):
            return {k: self._clip_observation(v) for k, v in obs.items()}
        elif isinstance(obs, np.ndarray):
            # Clip values to slightly inside [-1, 1] to be safe against float precision errors
            return np.clip(obs, -1.0, 1.0)
        return obs

    def reset(self, seed=None, options=None):
        obs, infos = self.env.reset(seed=seed, options=options)
        self.prev_min_dists = {}
        self.prev_zombie_count = 0
        
        # 2. Clip observation on reset
        obs = self._clip_observation(obs)
        return obs, infos

    def render(self, *args, **kwargs):
        return self.env.render()
    
    def _get_base_game(self):
        current_env = self.env
        for _ in range(20):
            if hasattr(current_env, "zombies") and hasattr(current_env, "agents"):
                return current_env
            if hasattr(current_env, "aec_env"):
                current_env = current_env.aec_env
            elif hasattr(current_env, "env"):
                current_env = current_env.env
            elif hasattr(current_env, "par_env"):
                current_env = current_env.par_env
            elif hasattr(current_env, "unwrapped") and current_env != current_env.unwrapped:
                current_env = current_env.unwrapped
            else:
                break     
        raise AttributeError("CRITICAL: Underlying KAZ environment not found!")

    def step(self, action_dict):
        obs, rewards, terminations, truncations, infos = self.env.step(action_dict)

        obs = self._clip_observation(obs)
        
        try:
            base_game = self._get_base_game()
        except AttributeError:
            return obs, rewards, terminations, truncations, infos

        shaped_rewards = rewards.copy()
        
        current_agents_obj = {a.name: a for a in base_game.agents}
        current_zombies = base_game.zombies
        zombie_count = len(current_zombies)
        zombies_changed = (zombie_count != self.prev_zombie_count)

        # 충돌 감지용 가상 그룹
        zombie_group = base_game.zombie_list if hasattr(base_game, 'zombie_list') else []

        for agent_id, reward in rewards.items():
            # 이미 죽은 에이전트 패스
            if terminations.get(agent_id, False) or truncations.get(agent_id, False):
                continue
            
            # === [NEW] 기사 킬 보상 5배 강화 로직 ===
            # 기본 환경 보상(reward)이 1.0(킬)인지 확인하고, 기사라면 추가 보너스 지급
            if "knight" in agent_id and reward >= 1.0:
                # 기본 1.0 + 추가 4.0 = 총 5.0 (궁수의 5배)
                shaped_rewards[agent_id] += 4.0 

            agent_obj = current_agents_obj.get(agent_id)
            if not agent_obj: continue

        # for agent_id in rewards.keys():
        #     if terminations.get(agent_id, False) or truncations.get(agent_id, False):
        #         continue
            
        #     agent_obj = current_agents_obj.get(agent_id)
        #     if not agent_obj: continue

        #     # 현재 행동이 공격(Attack)인지 확인 (Action 5)
        #     current_action = action_dict.get(agent_id)
        #     is_attacking = (current_action == 5)

        #     if zombie_count > 0:
        #         ax, ay = agent_obj.rect.center
        #         a_angle = agent_obj.angle
                
        #         # 1. 가장 가까운 좀비 정보 계산
        #         dists = [math.hypot(ax - z.rect.centerx, ay - z.rect.centery) for z in current_zombies]
        #         min_dist = min(dists)
        #         closest_zombie = current_zombies[dists.index(min_dist)]
        #         zx, zy = closest_zombie.rect.center

        #         # 거리 변화량 (양수면 가까워짐, 음수면 멀어짐)
        #         dist_delta = 0
        #         if agent_id in self.prev_min_dists and not zombies_changed:
        #             dist_delta = self.prev_min_dists[agent_id] - min_dist
                
    

        #         # === [A] 기사 (Knight) 보상 로직 ===
        #         if "knight" in agent_id:
        #             # (1) 기본 이동 보상 (Walking)
        #             # 공격 안 하고 그냥 접근만 해도 아주 조금은 칭찬해줌 (길 찾기용)
        #             if dist_delta > 0:
        #                 shaped_rewards[agent_id] += dist_delta * 0.002

        #             # (2) [핵심 요청] 공격 키를 눌렀을 때의 거리 변화 보상
        #             if is_attacking:
        #                 # 공격 키를 눌렀는데, 직전보다 거리가 좁혀졌다? (전진 공격)
        #                 if dist_delta > 0:
        #                     # 걷기 보상보다 훨씬 큰 가중치 (10배 이상)
        #                     # "공격하면서 다가가는 것"을 적극 권장
        #                     shaped_rewards[agent_id] += dist_delta * 0.03
                        
        #                 # 공격 키를 눌렀는데 뒤로 가거나 멈춰있다? -> 보상 없음 (혹은 미세 페널티)
        #                 # 이는 "제자리 공격"이나 "도망가며 공격"을 방지함

        #         # === [B] 궁수 (Archer) 보상 로직 ===
        #         elif "archer" in agent_id:
        #             # (1) 조준 정렬 보상 (공격 시에만 체크)
        #             if is_attacking:
        #                 target_angle = math.atan2(zy - ay, zx - ax)
        #                 angle_diff = (a_angle - target_angle + math.pi) % (2 * math.pi) - math.pi
                        
        #                 # 전방 90도 내에 좀비가 있을 때
        #                 if abs(angle_diff) < (math.pi / 2):
        #                     # 화살 궤적(직선)과 좀비와의 수직 거리 오차
        #                     aim_error_dist = min_dist * abs(math.sin(angle_diff))
                            
        #                     MAX_TOLERANCE = 50.0 # 50픽셀 이내면 인정
        #                     if aim_error_dist < MAX_TOLERANCE:
        #                         # 정확할수록 점수 높음 (최대 0.1)
        #                         accuracy_reward = (1.0 - (aim_error_dist / MAX_TOLERANCE)) * 0.1
        #                         shaped_rewards[agent_id] += accuracy_reward

        #         # 공통: 현재 거리 저장
        #         self.prev_min_dists[agent_id] = min_dist

        #         # === [C] 충돌 방지 (안전 거리 유지) ===
        #         # 공격 보상을 받으려다 몸통 박치기 하는 것 방지
        #         # 킬 보상(+1.0)의 절반인 -0.5를 주어 확실히 회피 유도
        #         danger_zone = agent_obj.rect.inflate(10, 10)
        #         dummy = pygame.sprite.Sprite()
        #         dummy.rect = danger_zone
        #         if pygame.sprite.spritecollide(dummy, zombie_group, False):
        #             shaped_rewards[agent_id] -= 0.5 

        #         # === [D] 헛스윙(Whiff) 페널티 ===
        #         # 아무때나 공격키 누르는 것 방지
        #         if is_attacking:
        #             # 기사: 너무 멀리서(60px 밖) 휘두르면 감점
        #             if "knight" in agent_id and min_dist > 60:
        #                 shaped_rewards[agent_id] -= 0.005
        #             # 궁수: 조준 오차가 너무 큰데 쏘면 감점 (위에서 보상 못 받은 경우)
        #             elif "archer" in agent_id:
        #                 # (보상을 못 받는 것 자체가 손해이므로 페널티는 작게)
        #                 shaped_rewards[agent_id] -= 0.002

        self.prev_zombie_count = zombie_count
        return obs, shaped_rewards, terminations, truncations, infos