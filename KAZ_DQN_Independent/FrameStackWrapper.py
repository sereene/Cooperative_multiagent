import numpy as np
from collections import deque
from gymnasium import spaces
from pettingzoo.utils.wrappers import BaseParallelWrapper

class FrameStackWrapper(BaseParallelWrapper):
    def __init__(self, env, num_stack=3):
        super().__init__(env)
        self.num_stack = num_stack
        
        self.frames = {agent: deque(maxlen=num_stack) for agent in env.possible_agents}
        
        self.observation_spaces = {}
        for agent in env.possible_agents:
            obs_space = env.observation_space(agent)
            
            if isinstance(obs_space, spaces.Box):
                # [수정] 원본 관측값이 2D(16, 15) 등이더라도 1D로 평탄화(Flatten)하여 계산
                # 예: (16, 15) -> 240 -> 스택 3개 -> 720
                
                # 1. 원본 공간의 평탄화된 크기 계산
                flat_dim = np.prod(obs_space.shape)
                
                # 2. 스택된 새로운 크기 계산
                new_shape = (int(flat_dim * num_stack),)
                
                # 3. low, high 값도 평탄화 후 타일링
                low = np.tile(obs_space.low.flatten(), num_stack)
                high = np.tile(obs_space.high.flatten(), num_stack)
                
                self.observation_spaces[agent] = spaces.Box(
                    low=low,
                    high=high,
                    shape=new_shape,
                    dtype=obs_space.dtype
                )
            else:
                raise ValueError(f"Agent {agent} has non-Box observation space. Only Box (Vector) is supported.")

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def _get_stacked_obs(self, agent, obs):
        """
        큐에 저장된 프레임들을 1D로 평탄화한 뒤 이어 붙여 반환
        """
        assert len(self.frames[agent]) == self.num_stack
        
        # [수정] 저장된 각 프레임을 먼저 flatten() 한 뒤 연결
        # 2D (16, 15) -> 1D (240,) ... 이것들을 연결
        flat_frames = [frame.flatten() for frame in self.frames[agent]]
        
        # axis=0으로 연결하여 긴 1D 벡터 생성
        stacked_obs = np.concatenate(flat_frames, axis=0)
        
        return stacked_obs

    def reset(self, seed=None, options=None):
        obs, infos = self.env.reset(seed=seed, options=options)
        
        for agent in self.env.possible_agents:
            self.frames[agent].clear()
            if agent in obs:
                # 초기 상태 복제
                for _ in range(self.num_stack):
                    self.frames[agent].append(obs[agent])
        
        stacked_obs = {}
        for agent, observation in obs.items():
            stacked_obs[agent] = self._get_stacked_obs(agent, observation)
            
        return stacked_obs, infos

    def step(self, actions):
        obs, rewards, terminations, truncations, infos = self.env.step(actions)
        
        stacked_obs = {}
        for agent, observation in obs.items():
            self.frames[agent].append(observation)
            stacked_obs[agent] = self._get_stacked_obs(agent, observation)
            
        return stacked_obs, rewards, terminations, truncations, infos