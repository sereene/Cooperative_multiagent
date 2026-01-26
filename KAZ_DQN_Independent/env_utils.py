import supersuit as ss
from pettingzoo.butterfly import knights_archers_zombies_v10
import numpy as np
from gymnasium.spaces import Box 
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv

# [중요] 새로 만든 래퍼 임포트
from RewardShapingWrapper import RewardShapingWrapper

MAX_CYCLES = 900

class FixedParallelPettingZooEnv(ParallelPettingZooEnv):
    """
    이 클래스는 오직 기술적인 문제(Obs Space 범위, Terminations 포맷)만 해결합니다.
    """
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
        
        # PettingZoo 버전 호환성 처리 (반환값 개수 맞춤)
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
        
        # RLLib 요구사항: __all__ 키 추가
        terminations["__all__"] = any(terminations.values())
        truncations["__all__"] = any(truncations.values())

        return obs, rewards, terminations, truncations, infos

def env_creator(config=None):
    # 1. 기본 환경 생성
    env = knights_archers_zombies_v10.parallel_env(
        spawn_rate=50,       
        num_archers=1,       
        num_knights=1,       
        max_arrows=1,      
        max_cycles=MAX_CYCLES,
        vector_state=True,   
        render_mode="rgb_array"
    )    
    # 3. [중요] 보상 셰이핑 래퍼 적용 (여기서 보상 로직이 추가됨)
    env = RewardShapingWrapper(env)

    env = FixedParallelPettingZooEnv(env)
    
    return env