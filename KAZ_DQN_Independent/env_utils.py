import supersuit as ss
from pettingzoo.butterfly import knights_archers_zombies_v10
import numpy as np
from gymnasium.spaces import Box 
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from FrameStackWrapper import FrameStackWrapper

from RewardShapingWrapper import RewardShapingWrapper

MAX_CYCLES = 900

class FixedParallelPettingZooEnv(ParallelPettingZooEnv):
    def __init__(self, env):
        # Obs Space를 무한대로 확장 (RLLib 호환성)
        for agent_id in env.possible_agents:
            obs_space = env.observation_spaces[agent_id]
            
            if isinstance(obs_space, Box):
                env.observation_spaces[agent_id] = Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=obs_space.shape,
                    dtype=obs_space.dtype
                )

        super().__init__(env)
        
    def reset(self, *, seed=None, options=None):
        return self.par_env.reset(seed=seed, options=options)

    def step(self, action_dict):
        step_result = self.par_env.step(action_dict)
        
        if len(step_result) == 5:
            obs, rewards, terminations, truncations, infos = step_result
        elif len(step_result) == 4:
            obs, rewards, dones, infos = step_result
            terminations = dones
            truncations = {agent: False for agent in self.par_env.agents}
        else:
            raise ValueError(f"Environment returned {len(step_result)} values.")
        
        if any(terminations.values()):
            for agent in terminations:
                terminations[agent] = True
        
        terminations["__all__"] = any(terminations.values())
        truncations["__all__"] = any(truncations.values())

        return obs, rewards, terminations, truncations, infos

def env_creator(config=None):
    # [수정] 기사 2명, 궁수 0명 설정
    env = knights_archers_zombies_v10.parallel_env(
        spawn_rate=50,       
        num_archers=0,       # [변경] 1 -> 0
        num_knights=2,       # [변경] 1 -> 2
        max_arrows=1,      
        max_cycles=MAX_CYCLES,
        vector_state=True,   
        render_mode="rgb_array"
    )    
    # env = RewardShapingWrapper(env)

    env = FrameStackWrapper(env, num_stack=3)

    env = FixedParallelPettingZooEnv(env)
    
    return env