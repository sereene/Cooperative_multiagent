import supersuit as ss
from pettingzoo.butterfly import cooperative_pong_v5
from ray.rllib.env.multi_agent_env import MultiAgentEnv
import numpy as np
from gymnasium import spaces


MAX_CYCLES = 900

class FixedParallelPettingZooEnv(MultiAgentEnv):
    """
    [Final Fix]
    Observation Space를 (84, 168, 1)로 정의했으므로,
    실제 step()과 reset()에서 들어오는 (84, 168) 데이터를 (84, 168, 1)로 변환해서 반환해야 합니다.
    """
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
        """
        관측 데이터가 2차원(H, W)인 경우 (H, W, 1)로 차원을 확장합니다.
        """
        for agent_id, obs in obs_dict.items():
            # numpy 배열이고 2차원이라면 차원 추가
            if isinstance(obs, np.ndarray) and obs.ndim == 2:
                obs_dict[agent_id] = np.expand_dims(obs, axis=-1)
        return obs_dict

    @property
    def possible_agents(self):
        return self.env.possible_agents

    def reset(self, *, seed=None, options=None):
        obs, infos = self.env.reset(seed=seed, options=options)
        # [수정] 데이터 차원 보정 후 반환
        return self._process_obs(obs), infos

    def step(self, action_dict):
        obs, rewards, terms, truncs, infos = self.env.step(action_dict)
        
        # [수정] 데이터 차원 보정
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
    env = ss.resize_v1(env, x_size=168, y_size=84)
    env = ss.color_reduction_v0(env, mode="full")
    
    # 3. Reward Shaping
    # env = RewardShapingWrapper(env)

    # 4. Frame Stacking (여기서 3을 썼으므로 3채널이 됩니다)
    # env = FrameStackWrapper(env, num_stack=3)

    # 5. RLLib 포장
    return FixedParallelPettingZooEnv(env)