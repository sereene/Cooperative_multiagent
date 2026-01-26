import numpy as np
from collections import deque
from gymnasium import spaces
from pettingzoo.utils.wrappers import BaseParallelWrapper

class FrameStackWrapper(BaseParallelWrapper):
<<<<<<< HEAD
    def __init__(self, env, num_stack=4, fade_decay=0.7):
        super().__init__(env)
        self.num_stack = num_stack
        self.fade_decay = fade_decay
        
        self.frames = {agent: deque(maxlen=num_stack) for agent in env.possible_agents}
        self.weights = [self.fade_decay**(num_stack - 1 - i) for i in range(num_stack)]
        
        # 목표 형태: (84, 168, 4)
        self.h = 84
        self.w = 168
        self.c = 1 # 흑백 이미지 가정
        self.new_channels = self.c * num_stack # 4
        
        # Space 정의
=======
    def __init__(self, env, num_stack=4):
        super().__init__(env)
        self.num_stack = num_stack
        
        self.frames = {agent: deque(maxlen=num_stack) for agent in env.possible_agents}
        
        self.h = 84
        self.w = 168
        self.c = 1 
        self.new_channels = self.c * num_stack # 4
        
>>>>>>> 412078245fd837dfbb4a3cb48757d339382f2fbd
        self.target_space = spaces.Box(
            low=0, 
            high=255, 
            shape=(self.h, self.w, self.new_channels), 
            dtype=np.uint8
        )
        
<<<<<<< HEAD
        # PettingZoo 호환성 업데이트
=======
>>>>>>> 412078245fd837dfbb4a3cb48757d339382f2fbd
        for agent in env.possible_agents:
            self.observation_spaces[agent] = self.target_space

    def observation_space(self, agent):
        return self.target_space

    def _process_obs(self, obs):
<<<<<<< HEAD
        """
        입력 관측값을 무조건 (84, 168, 1)로 변환
        """
        # 1. 2D (84, 168) -> (84, 168, 1)
        if obs.ndim == 2:
            return obs[..., None]
        
        # 2. 3D인데 차원이 이상한 경우
=======
        
        if obs.ndim == 2:
            return obs[..., None]
        
>>>>>>> 412078245fd837dfbb4a3cb48757d339382f2fbd
        # (1, 84, 168) -> (84, 168, 1)
        if obs.shape == (1, 84, 168):
            return np.transpose(obs, (1, 2, 0))
            
<<<<<<< HEAD
        # 3. 이미 (84, 168, 1)이면 그대로 반환
        return obs

    def _get_stacked_obs(self, agent, obs):
        # 1. 단일 프레임 정규화
        obs = self._process_obs(obs)
        
        # 2. 큐 업데이트
=======
        # 이미 (84, 168, 1)이면 그대로 반환
        return obs

    def _get_stacked_obs(self, agent, obs):
        obs = self._process_obs(obs)
        
>>>>>>> 412078245fd837dfbb4a3cb48757d339382f2fbd
        self.frames[agent].append(obs)
        while len(self.frames[agent]) < self.num_stack:
            self.frames[agent].append(obs)
        
<<<<<<< HEAD
        # 3. Fading 효과 적용
        faded_frames = []
        for i, frame in enumerate(self.frames[agent]):
            weight = self.weights[i]
            if weight == 1.0:
                faded_frames.append(frame)
            else:
                faded_frame = (frame.astype(np.float32) * weight).astype(np.uint8)
                faded_frames.append(faded_frame)
        
        # 4. 합치기 (84, 168, 4)
        stacked_obs = np.concatenate(faded_frames, axis=-1)
=======
        stacked_obs = np.concatenate(list(self.frames), axis=-1)
>>>>>>> 412078245fd837dfbb4a3cb48757d339382f2fbd
        
        return np.ascontiguousarray(stacked_obs, dtype=np.uint8)

    def reset(self, seed=None, options=None):
        obs, infos = self.env.reset(seed=seed, options=options)
        for agent in self.frames:
            self.frames[agent].clear()
        
        stacked_obs = {}
        for agent, observation in obs.items():
            stacked_obs[agent] = self._get_stacked_obs(agent, observation)
        return stacked_obs, infos

    def step(self, actions):
        obs, rewards, terminations, truncations, infos = self.env.step(actions)
        
        stacked_obs = {}
        for agent, observation in obs.items():
            stacked_obs[agent] = self._get_stacked_obs(agent, observation)
<<<<<<< HEAD
        return stacked_obs, rewards, terminations, truncations, infos
=======
        return stacked_obs, rewards, terminations, truncations, infos
>>>>>>> 412078245fd837dfbb4a3cb48757d339382f2fbd
