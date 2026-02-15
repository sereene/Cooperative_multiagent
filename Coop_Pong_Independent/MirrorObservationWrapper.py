import numpy as np

class MirrorObservationWrapper:
    def __init__(self, env):
        self.env = env
        self.possible_agents = env.possible_agents
        
        # 오른쪽 에이전트(paddle_1) 식별을 위한 인덱스 찾기
        # 보통 ['paddle_0', 'paddle_1'] 순서임
        try:
            self.right_agent_idx = self.possible_agents.index("paddle_1")
            self.right_agent_id = "paddle_1"
        except ValueError:
            # 만약 이름이 다르다면 두 번째 에이전트를 오른쪽으로 가정
            self.right_agent_idx = 1
            self.right_agent_id = self.possible_agents[1]

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        return self._flip_obs(obs), info

    def step(self, actions):
        obs, rewards, terms, truncs, infos = self.env.step(actions)
        return self._flip_obs(obs), rewards, terms, truncs, infos

    def _flip_obs(self, obs_dict):
        # 원본 딕셔너리를 건드리지 않기 위해 복사본 생성 권장
        new_obs = obs_dict.copy()
        
        if self.right_agent_id in new_obs:
            # 오른쪽 에이전트의 이미지를 좌우 반전 (Horizontal Flip)
            # obs shape: (Height, Width, Channel) -> Width는 index 1
            # np.fliplr는 2차원 이상의 배열에서 2번째 차원(가로)을 뒤집음
            new_obs[self.right_agent_id] = np.fliplr(new_obs[self.right_agent_id])
            
            # 주의: np.fliplr은 메모리 stride가 꼬일 수 있어 .copy()로 연속 메모리 확보가 안전함
            new_obs[self.right_agent_id] = np.ascontiguousarray(new_obs[self.right_agent_id])
            
        return new_obs

    # 아래 메서드들은 래퍼로서 그대로 전달
    def __getattr__(self, name):
        return getattr(self.env, name)