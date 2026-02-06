import shimmy
from shimmy import MeltingPotCompatibilityV0
from shimmy.utils.meltingpot import load_meltingpot
import numpy as np
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv

# [주의] 파일명 대소문자 확인
try:
    from RewardShapingWrapper import RewardShapingWrapper
except ImportError:
    from rewardshaping_wrapper import RewardShapingWrapper

class FixedParallelPettingZooEnv(ParallelPettingZooEnv):
    """
    Shimmy/MeltingPot <-> RLlib 호환성 래퍼
    """
    def __init__(self, env):
        super().__init__(env)
        # [수정됨] RLlib 에러 해결을 위해 _agent_ids 명시
        # Shimmy/PettingZoo는 static한 에이전트 목록을 possible_agents에 보관합니다.
        self._agent_ids = set(self.par_env.possible_agents)

    def reset(self, *, seed=None, options=None):
        return self.par_env.reset(seed=seed, options=options)

    def step(self, action_dict):
        step_result = self.par_env.step(action_dict)
        
        if len(step_result) == 5:
            obs, rewards, terminations, truncations, infos = step_result
        else:
            obs, rewards, dones, infos = step_result
            terminations = dones
            truncations = {a: False for a in self.par_env.agents}
            
        terminations["__all__"] = any(terminations.values())
        truncations["__all__"] = any(truncations.values())
        return obs, rewards, terminations, truncations, infos

def env_creator(config=None):
    # 1. Melting Pot 기저 환경 로드
    substrate_name = "commons_harvest__partnership"
    substrate = load_meltingpot(substrate_name)
    
    # 2. PettingZoo 변환
    env = MeltingPotCompatibilityV0(substrate, render_mode="rgb_array")
    
    # 3. 보상 셰이핑 적용
    env = RewardShapingWrapper(env, sharing_coeff=0.5)

    # 4. RLlib 호환 래퍼 적용
    env = FixedParallelPettingZooEnv(env)
    
    return env