import numpy as np
import pygame
from pettingzoo.utils.wrappers import BaseParallelWrapper

class RewardShapingWrapper(BaseParallelWrapper):

    def __init__(self, env):
        super().__init__(env)

        # CooperativePong 객체 가져오기
        try:
            self.game = self.env.env.env
        except:
            self.game = self.env.unwrapped.env

        # 1. 화면 크기 가져오기
        self.screen_width = getattr(self.game, 's_width', 480)
        self.screen_height = getattr(self.game, 's_height', 280)
        
        # 2. 기준 거리 설정
        # Termination(실점) 시에는 공과 패들의 X좌표가 거의 같으므로,
        # Y축 차이가 중요--> 정규화 기준을 '화면 높이'로 설정
        self.max_dist_threshold = float(self.screen_height)

    def _get_center(self, obj):
        """객체의 중심 좌표를 반환하는 헬퍼 함수"""
        if hasattr(obj, "rects"):
            return np.array(obj.rects[0].center)
        return np.array(obj.rect.center)

    def reset(self, seed=None, options=None):
        return self.env.reset(seed=seed, options=options)

    def step(self, actions):
        obs, rewards, terminations, truncations, infos = self.env.step(actions)

        ball_rect = self.game.ball.rect
        ball_center = np.array(ball_rect.center)
        screen_center_x = self.screen_width / 2
        
        paddle_objs = {
            self.game.agents[0]: self.game.p0,
            self.game.agents[1]: self.game.p1
        }
        
        paddles_rect = {}
        for agent_id, p_obj in paddle_objs.items():
            if hasattr(p_obj, "rects"):
                paddles_rect[agent_id] = p_obj.rects[0]
            else:
                paddles_rect[agent_id] = p_obj.rect
        
        for i, agent in enumerate(self.env.agents):
            if agent in rewards:
                # 1. 충돌(Hit) 보상 (0.5)
                if ball_rect.colliderect(paddles_rect[agent]):
                    rewards[agent] += 0.5
                
                # 2. 매 스텝 거리 보상 (최대 0.01)
                # 에이전트 인덱스 확인 (0: 왼쪽, 1: 오른쪽 가정)
                is_left_agent = (i == 0)
                
                # 공이 해당 에이전트의 구역에 있을 때만 거리 보상 계산
                # (왼쪽 에이전트는 공이 화면 절반보다 왼쪽에 있을 때, 오른쪽은 그 반대)
                ball_in_my_zone = (ball_center[0] < screen_center_x) if is_left_agent else (ball_center[0] >= screen_center_x)

                if ball_in_my_zone:
                    paddle_center = self._get_center(paddle_objs[agent])
                    
                    # Y축 거리 계산
                    vertical_distance = abs(ball_center[1] - paddle_center[1])
                    
                    # 매 스텝 보상은 충돌 보상(0.5)보다 훨씬 작아야 함
                    # 안 그러면 공을 안 치고 따라다니기만 하는 게 이득이라고 판단함
                    step_max_reward = 0.01  
                    
                    if vertical_distance < self.screen_height:
                        # 정규화 (0 ~ 1 사이 값)
                        normalized_dist = vertical_distance / self.screen_height
                        
                        # 거리가 0에 가까울수록 step_max_reward(0.01)에 가까워짐
                        dist_reward = step_max_reward * (1.0 - normalized_dist)
                        
                        rewards[agent] += dist_reward

        # # 2. 에피소드 종료(Termination) 시 거리 기반 보상

        # is_terminated = any(terminations.values())

        # if is_terminated:
        #     ball_center = np.array(ball_rect.center)
        #     screen_center_x = self.screen_width / 2
            
        #     responsible_agent = None

        #     # 공이 어느 쪽 구역에 있는지를 기준으로 책임 에이전트 선정
        #     if ball_center[0] < screen_center_x:
        #         responsible_agent = self.env.possible_agents[0] # 왼쪽 에이전트
        #     else:
        #         responsible_agent = self.env.possible_agents[1] # 오른쪽 에이전트
            
        #     # 책임자가 있고 보상 딕셔너리에 존재할 경우
        #     if responsible_agent and responsible_agent in rewards:
        #         paddle_center = self._get_center(paddle_objs[responsible_agent])
                
        #         # 거리 계산 (Termination 시에는 사실상 Y축 거리 차이와 유사)
        #         dist = np.linalg.norm(ball_center - paddle_center)
                
        #         max_reward = 0.5
                
        #         # 기준 거리(screen_height)보다 가까울 때만 보상 지급
        #         if dist < self.screen_height:
        #             # 거리가 0에 가까울수록 max_reward, 멀어지면 0으로 선형 감소
        #             dist_reward = max_reward * (1.0 - (dist / self.max_dist_threshold))
        #             rewards[responsible_agent] += dist_reward

        return obs, rewards, terminations, truncations, infos
