import supersuit as ss
from pettingzoo.butterfly import cooperative_pong_v5
from ray.rllib.env.multi_agent_env import MultiAgentEnv
import numpy as np
from gymnasium import spaces
from FrameStackWrapper import FrameStackWrapper


MAX_CYCLES = 900

class FixedParallelPettingZooEnv(MultiAgentEnv):

    def __init__(self, pettingzoo_env):
        super().__init__()
        self.env = pettingzoo_env
        self._agent_ids = set(self.env.possible_agents)
        
        sample_agent = self.env.possible_agents[0]
        original_obs_space = self.env.observation_space(sample_agent)
        
        # 1. 공간 정의 (Shape definition)
        if len(original_obs_space.shape) == 3:
            final_shape = original_obs_space.shape
        else:
            # 2D 데이터(84, 168)라면 (84, 168, 1)로 정의
            final_shape = original_obs_space.shape + (1,)

        self.observation_space = spaces.Box(
            low=0, 
            high=255, 
            shape=final_shape, 
            dtype=np.uint8
        )
        
        self.action_space = self.env.action_space(sample_agent)

        # 게임 객체 확보
        try:
            self.game = self.env.unwrapped.env
        except:
            try: self.game = self.env.env.env
            except: self.game = None
            
        if self.game:
            self.screen_width = float(getattr(self.game, 's_width', 480))
            self.screen_height = float(getattr(self.game, 's_height', 280))
        else:
            self.screen_width = 480.0
            self.screen_height = 280.0

    def _process_obs(self, obs_dict):
        for agent_id, obs in obs_dict.items():
            if isinstance(obs, np.ndarray):
                # 1. 2차원 흑백 데이터 (84, 84)가 정상적으로 들어온 경우 -> (84, 84, 1)
                if obs.ndim == 2:
                    obs_dict[agent_id] = np.expand_dims(obs, axis=-1)
                
                # 2. 3차원 RGB 데이터 (84, 84, 3)가 그대로 들어온 경우 강제 변환
                elif obs.ndim == 3 and obs.shape[-1] == 3:
                    # RGB를 Grayscale로 변환 (표준 휘도 공식 적용)
                    gray = np.dot(obs[..., :3], [0.2989, 0.5870, 0.1140])
                    # (84, 84) 형태의 배열을 (84, 84, 1)로 확장하고 uint8로 캐스팅
                    obs_dict[agent_id] = np.expand_dims(gray.astype(np.uint8), axis=-1)
                    
        return obs_dict

    @property
    def possible_agents(self):
        return self.env.possible_agents

    def reset(self, *, seed=None, options=None):
        obs, infos = self.env.reset(seed=seed, options=options)
        return self._process_obs(obs), infos

    def step(self, action_dict):
        obs, rewards, terms, truncs, infos = self.env.step(action_dict)
        
        obs = self._process_obs(obs)
        
        terms["__all__"] = any(terms.values())
        truncs["__all__"] = any(truncs.values())
        return obs, rewards, terms, truncs, infos

    def render(self):
        return self.env.render()
    
    def close(self):
        return self.env.close()

def env_creator(config=None):
    # 1. PettingZoo 환경 생성
    env = cooperative_pong_v5.parallel_env(max_cycles=MAX_CYCLES, render_mode="rgb_array")

    # 2. SuperSuit Wrappers
    env = ss.resize_v1(env, x_size=84, y_size=84)
    env = ss.color_reduction_v0(env, mode="full")
    
    # 3. Reward Shaping
    # env = RewardShapingWrapper(env)

    # 4. Frame Stacking 
    # env = FrameStackWrapper(env, num_stack=3)

    # 5. RLLib 포장
    return FixedParallelPettingZooEnv(env)
