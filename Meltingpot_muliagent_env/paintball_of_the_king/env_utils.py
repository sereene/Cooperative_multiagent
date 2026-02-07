# from framestack import FrameStackWrapper
# import shimmy
# from shimmy import MeltingPotCompatibilityV0
# from shimmy.utils.meltingpot import load_meltingpot
# import numpy as np
# from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
# from pettingzoo.utils.wrappers import BaseParallelWrapper
# import gymnasium as gym  



#  # [추가] RGB 이미지만 추출하는 래퍼
# class MeltingPotRGBWrapper(BaseParallelWrapper):
#     def __init__(self, env):
#         super().__init__(env)
#         # 모든 에이전트의 관측 공간을 RGB Box로 변경
#         self.observation_spaces = {}
#         # 가능한 모든 에이전트 목록 확보
#         agents = getattr(env, "possible_agents", getattr(env, "agents", []))
        
#         for agent in agents:
#             # 원본 공간이 Dict라고 가정하고 RGB 키의 공간을 가져옴
#             # (Shimmy는 기본적으로 Dict 공간을 반환함)
#             obs_space = env.observation_space(agent)
#             if isinstance(obs_space, gym.spaces.Dict):
#                 self.observation_spaces[agent] = obs_space["RGB"]
#             else:
#                 # 이미 Box라면 그대로 사용 (안전장치)
#                 self.observation_spaces[agent] = obs_space

#     def reset(self, seed=None, options=None):
#         obs, infos = self.env.reset(seed=seed, options=options)
#         return self._process_obs(obs), infos

#     def step(self, actions):
#         obs, rewards, terminations, truncations, infos = self.env.step(actions)
#         return self._process_obs(obs), rewards, terminations, truncations, infos

#     def _process_obs(self, obs_dict):
#         # 딕셔너리 관측값에서 'RGB'만 꺼내서 반환
#         new_obs = {}
#         for agent, data in obs_dict.items():
#             if isinstance(data, dict) and "RGB" in data:
#                 new_obs[agent] = data["RGB"]
#             else:
#                 new_obs[agent] = data
#         return new_obs

# class ShimmyCompatibilityWrapper(BaseParallelWrapper):
#     """
#     1. Shimmy의 observation_space 메서드 에러 해결
#     2. RLlib이 일부 에이전트의 Action만 보낼 때 발생하는 KeyError 해결
#     """
#     def __init__(self, env):
#         super().__init__(env)
        
#         self.observation_spaces = {}
#         self.action_spaces = {}
        
#         # Shimmy/PettingZoo 환경의 모든 에이전트 리스트 가져오기
#         # possible_agents가 없으면 agents를 사용
#         agent_list = getattr(env, "possible_agents", getattr(env, "agents", []))
#         self._all_agents_set = set(agent_list) # 빠른 조회를 위해 set 저장
        
#         for agent in agent_list:
#             # 1. Observation Space 구축
#             try:
#                 if hasattr(env, "observation_space"):
#                     # callable이면 호출, 아니면 그대로 사용
#                     raw = env.observation_space
#                     self.observation_spaces[agent] = raw(agent) if callable(raw) else raw
#                 elif hasattr(env, "observation_spaces"):
#                     self.observation_spaces[agent] = env.observation_spaces[agent]
#             except Exception as e:
#                 print(f"[Warning] Failed to get observation_space for {agent}: {e}")
#                 # 최후의 수단: dict 속성 접근
#                 if hasattr(env, "observation_space") and isinstance(env.observation_space, dict):
#                     self.observation_spaces[agent] = env.observation_space[agent]

#             # 2. Action Space 구축
#             try:
#                 if hasattr(env, "action_space"):
#                     raw = env.action_space
#                     self.action_spaces[agent] = raw(agent) if callable(raw) else raw
#                 elif hasattr(env, "action_spaces"):
#                     self.action_spaces[agent] = env.action_spaces[agent]
#             except Exception as e:
#                 print(f"[Warning] Failed to get action_space for {agent}: {e}")
#                 if hasattr(env, "action_space") and isinstance(env.action_space, dict):
#                     self.action_spaces[agent] = env.action_space[agent]

#     def reset(self, seed=None, options=None):
#         return self.env.reset(seed=seed, options=options)

#     def step(self, actions):
#         # [핵심 수정] RLlib이 일부 에이전트의 액션만 보낼 경우(KeyError 방지)
#         # Shimmy는 모든 에이전트의 액션을 리스트로 요구하므로, 누락된 에이전트는 NOOP(0) 처리
        
#         full_actions = actions.copy()
        
#         # 현재 환경에서 요구하는 에이전트 목록 (가능한 모든 에이전트)
#         target_agents = getattr(self.env, "possible_agents", self.env.agents)
        
#         for agent in target_agents:
#             if agent not in full_actions:
#                 # 0번 액션은 보통 NOOP(행동 없음)입니다.
#                 full_actions[agent] = 0 
        
#         return self.env.step(full_actions)

# class FixedParallelPettingZooEnv(ParallelPettingZooEnv):
#     """Shimmy <-> RLlib 호환성 래퍼"""
#     def __init__(self, env):
#         super().__init__(env)
#         self._agent_ids = set(self.par_env.possible_agents)

#     def reset(self, *, seed=None, options=None):
#         return self.par_env.reset(seed=seed, options=options)

#     def step(self, action_dict):
#         step_result = self.par_env.step(action_dict)
#         if len(step_result) == 5:
#             obs, rewards, terminations, truncations, infos = step_result
#         else:
#             obs, rewards, dones, infos = step_result
#             terminations = dones
#             truncations = {a: False for a in self.par_env.agents}
        
#         terminations["__all__"] = any(terminations.values())
#         truncations["__all__"] = any(truncations.values())
#         return obs, rewards, terminations, truncations, infos

# def env_creator(config=None):
#     if config is None: config = {}
    
#     # 1. Substrate 로드
#     substrate_name = config.get("substrate", "paintball__king_of_the_hill")
#     substrate = load_meltingpot(substrate_name)
    
#     # 2. Shimmy 변환
#     env = MeltingPotCompatibilityV0(substrate, render_mode="rgb_array")

#     env = MeltingPotRGBWrapper(env)

#     env = FrameStackWrapper(env, num_stack=3)
    
#     # 3. [에러 해결] 호환성 래퍼 (KeyError 및 Obs Space 오류 해결)
#     env = ShimmyCompatibilityWrapper(env)
        
#     # 5. RLlib 호환성 래퍼
#     env = FixedParallelPettingZooEnv(env)
#     return env

import gymnasium as gym
import shimmy
from shimmy import MeltingPotCompatibilityV0
from shimmy.utils.meltingpot import load_meltingpot
import numpy as np
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from pettingzoo.utils.wrappers import BaseParallelWrapper

from framestack import FrameStackWrapper 

class MeltingPotRGBWrapper(BaseParallelWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_spaces = {}
        agents = getattr(env, "possible_agents", getattr(env, "agents", []))
        
        for agent in agents:
            obs_space = env.observation_space(agent)
            if isinstance(obs_space, gym.spaces.Dict):
                self.observation_spaces[agent] = obs_space["RGB"]
            else:
                self.observation_spaces[agent] = obs_space

    # [핵심 추가] 이 메서드가 있어야 다음 래퍼가 "아, 이건 Box구나"라고 인식함
    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def reset(self, seed=None, options=None):
        obs, infos = self.env.reset(seed=seed, options=options)
        return self._process_obs(obs), infos

    def step(self, actions):
        obs, rewards, terminations, truncations, infos = self.env.step(actions)
        return self._process_obs(obs), rewards, terminations, truncations, infos

    def _process_obs(self, obs_dict):
        new_obs = {}
        for agent, data in obs_dict.items():
            if isinstance(data, dict) and "RGB" in data:
                new_obs[agent] = data["RGB"]
            else:
                new_obs[agent] = data
        return new_obs

class ShimmyCompatibilityWrapper(BaseParallelWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_spaces = {}
        self.action_spaces = {}
        agent_list = getattr(env, "possible_agents", getattr(env, "agents", []))
        
        for agent in agent_list:
            # 안전하게 공간 가져오기
            try:
                # env.observation_space(agent) 호출 시 이전 래퍼들의 오버라이딩된 메서드가 호출됨
                if hasattr(env, "observation_space"):
                    raw = env.observation_space
                    self.observation_spaces[agent] = raw(agent) if callable(raw) else raw
                elif hasattr(env, "observation_spaces"):
                    self.observation_spaces[agent] = env.observation_spaces[agent]
            except Exception:
                # 실패 시 최후의 수단
                if hasattr(env, "observation_space") and isinstance(env.observation_space, dict):
                    self.observation_spaces[agent] = env.observation_space[agent]

            try:
                if hasattr(env, "action_space"):
                    raw = env.action_space
                    self.action_spaces[agent] = raw(agent) if callable(raw) else raw
                elif hasattr(env, "action_spaces"):
                    self.action_spaces[agent] = env.action_spaces[agent]
            except Exception:
                if hasattr(env, "action_space") and isinstance(env.action_space, dict):
                    self.action_spaces[agent] = env.action_space[agent]

    # [중요] 여기서도 오버라이딩
    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def reset(self, seed=None, options=None):
        return self.env.reset(seed=seed, options=options)

    def step(self, actions):
        full_actions = actions.copy()
        target_agents = getattr(self.env, "possible_agents", self.env.agents)
        for agent in target_agents:
            if agent not in full_actions:
                full_actions[agent] = 0 
        return self.env.step(full_actions)

class FixedParallelPettingZooEnv(ParallelPettingZooEnv):
    def __init__(self, env):
        super().__init__(env)
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
    if config is None: config = {}
    
    substrate_name = config.get("substrate", "paintball__king_of_the_hill")
    substrate = load_meltingpot(substrate_name)
    
    # 1. Shimmy 변환
    env = MeltingPotCompatibilityV0(substrate, render_mode="rgb_array")
    
    # 2. RGB만 남기기 (Dict -> Box)
    env = MeltingPotRGBWrapper(env)

    # 3. 프레임 스택 (Box -> Box 12ch)
    env = FrameStackWrapper(env, num_stack=4)
    
    # 4. 호환성 래퍼
    env = ShimmyCompatibilityWrapper(env)
        
    # 5. RLlib 호환성 래퍼
    env = FixedParallelPettingZooEnv(env)
    return env
