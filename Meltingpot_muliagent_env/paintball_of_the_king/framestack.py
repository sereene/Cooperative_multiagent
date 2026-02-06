import numpy as np
from collections import deque
from gymnasium import spaces
from pettingzoo.utils.wrappers import BaseParallelWrapper

class FrameStackWrapper(BaseParallelWrapper):
    def __init__(self, env, num_stack=4):
        super().__init__(env)
        self.num_stack = num_stack
        
        self.frames = {agent: deque(maxlen=num_stack) for agent in env.possible_agents}
        
        self.h = 88
        self.w = 88
        self.c = 3 
        self.new_channels = self.c * num_stack 
        
        # [수정] 관측 공간을 Dict가 아닌 Box(이미지)로 통일
        self.observation_spaces = {}
        for agent in env.possible_agents:
            self.observation_spaces[agent] = spaces.Box(
                low=0, 
                high=255, 
                shape=(self.h, self.w, self.new_channels), 
                dtype=np.uint8
            )

    # [안전장치] PettingZoo가 이 함수를 호출할 때 올바른 공간을 반환하도록 강제
    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def _get_stacked_obs(self, agent, observation):
        # 입력 처리 (Dict -> Array)
        if isinstance(observation, dict) and 'RGB' in observation:
            rgb_frame = observation['RGB']
        else:
            rgb_frame = observation

        # 큐에 추가
        self.frames[agent].append(rgb_frame)
        
        # 스택 반환
        return np.concatenate(list(self.frames[agent]), axis=-1)

    def reset(self, seed=None, options=None):
        obs, infos = self.env.reset(seed=seed, options=options)
        
        for agent, observation in obs.items():
            # [핵심 수정] 리셋 시에는 첫 프레임으로 버퍼를 '가득' 채워야 함
            if isinstance(observation, dict) and 'RGB' in observation:
                rgb_frame = observation['RGB']
            else:
                rgb_frame = observation
            
            # 버퍼 초기화 후 num_stack 만큼 복사해서 채움
            self.frames[agent].clear()
            for _ in range(self.num_stack):
                self.frames[agent].append(rgb_frame)
            
            # 스택 결과로 덮어쓰기
            obs[agent] = np.concatenate(list(self.frames[agent]), axis=-1)
            
        return obs, infos

    def step(self, actions):
        obs, rewards, terminations, truncations, infos = self.env.step(actions)
        
        for agent, observation in obs.items():
            obs[agent] = self._get_stacked_obs(agent, observation)
            
        return obs, rewards, terminations, truncations, infos