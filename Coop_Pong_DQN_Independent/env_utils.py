import supersuit as ss
from pettingzoo.butterfly import cooperative_pong_v5
from ray.rllib.env.multi_agent_env import MultiAgentEnv
import numpy as np
from gymnasium import spaces

# 사용자가 만든 Wrapper Import
from RewardShapingWrapper import RewardShapingWrapper
from FrameStackWrapper import FrameStackWrapper

MAX_CYCLES = 500

class FixedParallelPettingZooEnv(MultiAgentEnv):
    """
    [Final Fix]
    Observation Space를 동적으로 가져오는 대신, 
    우리가 목표로 하는 (84, 168, 4) 형태로 강제 고정(Hard-coding)합니다.
    이렇게 하면 중간 Wrapper들이 정보를 제대로 전달하지 못해도 RLLib은 3D로 인식합니다.
    """
    def __init__(self, pettingzoo_env):
        super().__init__()
        self.env = pettingzoo_env
        self._agent_ids = set(self.env.possible_agents)
        
        # [핵심 수정] 환경에 물어보지 않고, 우리가 아는 정답(84, 168, 4)을 강제로 박아넣습니다.
        # 이렇게 하면 "outside given space (84, 168)" 오류가 발생할 수 없습니다.
        self.observation_space = spaces.Box(
            low=0, 
            high=255, 
            shape=(84, 168, 4), 
            dtype=np.uint8
        )
        
        # Action Space는 그대로 가져옵니다.
        sample_agent = self.env.possible_agents[0]
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

    @property
    def possible_agents(self):
        return self.env.possible_agents

    def reset(self, *, seed=None, options=None):
        return self.env.reset(seed=seed, options=options)

    def step(self, action_dict):
        obs, rewards, terms, truncs, infos = self.env.step(action_dict)
        
        # Reward Shaping은 Wrapper에서 처리되므로 여기서는 패스
        
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
    env = RewardShapingWrapper(env)

    # 4. Frame Stacking
    env = FrameStackWrapper(env, num_stack=4)

    # 5. RLLib 포장
    return FixedParallelPettingZooEnv(env)