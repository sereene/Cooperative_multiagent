import numpy as np
import pygame
from pettingzoo.utils.wrappers import BaseParallelWrapper

class RewardShapingWrapper(BaseParallelWrapper):

    def __init__(self, env):
        super().__init__(env)

        # CooperativePong 객체 정확히 가져오기
        # parallel_env.env.env = CooperativePong 인스턴스
        try:
            self.game = self.env.env.env
        except:
            self.game = self.env.unwrapped.env

        self.previous_potentials = {}
        self.gamma = 0.99

    def _get_potential(self, agent_id):

        # 공 중심 좌표
        ball_center = np.array(self.game.ball.rect.center)

        # 패들 좌표 (CakePaddle 또는 Paddle 모두 지원)
        if agent_id == self.game.agents[0]:  # paddle_0
            paddle_obj = self.game.p0
        else:  # paddle_1
            paddle_obj = self.game.p1

        if hasattr(paddle_obj, "rects"):
            paddle_center = np.array(paddle_obj.rects[0].center)
        else:
            paddle_center = np.array(paddle_obj.rect.center)
        
        #잠재 함수(Potential) 계산: 공과 패들의 거리가 가까울수록 높은 값(0에 근접)을 가짐
        #Potential = -Distance * Scale

        # 공과 패들 간의 유클리드 거리 계산
        dist = np.linalg.norm(ball_center - paddle_center)

        return -dist * 0.01

    def reset(self, seed=None, options=None):
        obs, infos = self.env.reset(seed=seed, options=options)

        self.previous_potentials = {
            agent: self._get_potential(agent) for agent in self.env.agents
        }

        return obs, infos

    def step(self, actions):

        obs, rewards, terminations, truncations, infos = self.env.step(actions)

        # 공 위치 rect 가져오기
        ball_rect = self.game.ball.rect

        # paddle rects 가져오기
        if hasattr(self.game.p0, "rects"):
            paddle0_rect = self.game.p0.rects[0]
        else:
            paddle0_rect = self.game.p0.rect

        if hasattr(self.game.p1, "rects"):
            paddle1_rect = self.game.p1.rects[0]
        else:
            paddle1_rect = self.game.p1.rect

        paddles = {
            self.game.agents[0]: paddle0_rect,
            self.game.agents[1]: paddle1_rect
        }

        for agent in self.env.agents:

            if agent not in rewards: #reward가 없는 경우
                continue

            # potential based reward shaping 계산
            current_potential = self._get_potential(agent)
            prev_potential = self.previous_potentials.get(agent, current_potential)

            # 변화량 계산 (이전보다 가까워지면 +, 멀어지면 -)
            potential_reward = (self.gamma * current_potential) - prev_potential

            collision_reward = 0.0
            if ball_rect.colliderect(paddles[agent]):
                collision_reward = 0.5 # 패들과 공이 충돌 시 추가 보상

            # 충돌 보상과 거리 기반 잠재 보상 합산
            total_shaping = potential_reward + collision_reward

            # 최종 보상에 reward shaping 추가
            rewards[agent] += total_shaping

            # 이전 잠재 함수 값 업데이트
            self.previous_potentials[agent] = current_potential

        return obs, rewards, terminations, truncations, infos
