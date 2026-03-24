import supersuit as ss
from pettingzoo.butterfly import knights_archers_zombies_v10
import numpy as np
from gymnasium.spaces import Box 
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from FrameStackWrapper import FrameStackWrapper
from gymnasium.spaces import Tuple



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
    

def grouped_env_creator(config=None):
    """QMIX를 위해 에이전트들을 하나의 그룹으로 묶는 래퍼 함수"""
    env = env_creator(config)
    
    # 기사 2명을 "group_1" 이라는 하나의 엔티티로 묶습니다.
    grouping = {
        "group_1": ["knight_0", "knight_1"]
    }
    
    # [수정된 부분] 
    # RLlib 래퍼(env) 내부의 원본 PettingZoo 환경(par_env)에 접근해서
    # 우리가 수정한 무한대 Box 공간을 명시적으로 가져옵니다!
    obs_space = Tuple([
        env.par_env.observation_spaces["knight_0"], 
        env.par_env.observation_spaces["knight_1"]
    ])
    act_space = Tuple([
        env.par_env.action_spaces["knight_0"], 
        env.par_env.action_spaces["knight_1"]
    ])
    
    return env.with_agent_groups(grouping, obs_space=obs_space, act_space=act_space)


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

    env = FixedParallelPettingZooEnv(env)
    
    return env