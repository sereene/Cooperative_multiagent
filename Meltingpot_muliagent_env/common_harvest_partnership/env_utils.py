import gymnasium as gym
import shimmy
from shimmy import MeltingPotCompatibilityV0
from shimmy.utils.meltingpot import load_meltingpot
import numpy as np
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from pettingzoo.utils.wrappers import BaseParallelWrapper

# [설정] FrameStackWrapper 임포트 시도
try:
    from framestack import FrameStackWrapper
except ImportError:
    # 파일이 없으면 더미 클래스로 대체 (에러 방지용)
    class FrameStackWrapper(gym.Wrapper):
        def __init__(self, env, num_stack):
            super().__init__(env)
            self.num_stack = num_stack

class MeltingPotRGBWrapper(BaseParallelWrapper):
    """
    관측값(Observation) 딕셔너리에서 'RGB' 이미지만 추출하여
    모델이 처리할 수 있는 형태(Box)로 변환하는 래퍼
    """
    def __init__(self, env):
        super().__init__(env)
        self.observation_spaces = {}
        # 환경의 모든 에이전트 식별
        agents = getattr(env, "possible_agents", getattr(env, "agents", []))
        
        for agent in agents:
            obs_space = env.observation_space(agent)
            # Dict 공간이면 RGB만 추출
            if isinstance(obs_space, gym.spaces.Dict):
                self.observation_spaces[agent] = obs_space["RGB"]
            else:
                self.observation_spaces[agent] = obs_space

    # [핵심] 관측 공간을 Box(이미지)로 속이기
    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def reset(self, seed=None, options=None):
        obs, infos = self.env.reset(seed=seed, options=options)
        return self._process_obs(obs), infos

    def step(self, actions):
        obs, rewards, terminations, truncations, infos = self.env.step(actions)
        return self._process_obs(obs), rewards, terminations, truncations, infos

    def _process_obs(self, obs_dict):
        # 딕셔너리 {agent: {'RGB': ...}} -> {agent: RGB_Array} 변환
        new_obs = {}
        for agent, data in obs_dict.items():
            if isinstance(data, dict) and "RGB" in data:
                new_obs[agent] = data["RGB"]
            else:
                new_obs[agent] = data
        return new_obs

    # 렌더링 호환성
    def render(self, *args, **kwargs):
        return self.env.render()

class ShimmyCompatibilityWrapper(BaseParallelWrapper):
    """Shimmy와 RLlib 사이의 공간 정의(Space Definition) 호환성 해결"""
    def __init__(self, env):
        super().__init__(env)
        self.observation_spaces = {}
        self.action_spaces = {}
        agent_list = getattr(env, "possible_agents", getattr(env, "agents", []))
        
        for agent in agent_list:
            # Observation Space 안전하게 가져오기
            try:
                # 상위 래퍼(RGBWrapper)가 오버라이딩한 observation_space 메서드 호출
                if hasattr(env, "observation_space"):
                    raw = env.observation_space
                    self.observation_spaces[agent] = raw(agent) if callable(raw) else raw
                elif hasattr(env, "observation_spaces"):
                    self.observation_spaces[agent] = env.observation_spaces[agent]
            except Exception:
                if hasattr(env, "observation_space") and isinstance(env.observation_space, dict):
                    self.observation_spaces[agent] = env.observation_space[agent]

            # Action Space 가져오기
            try:
                if hasattr(env, "action_space"):
                    raw = env.action_space
                    self.action_spaces[agent] = raw(agent) if callable(raw) else raw
                elif hasattr(env, "action_spaces"):
                    self.action_spaces[agent] = env.action_spaces[agent]
            except Exception:
                if hasattr(env, "action_space") and isinstance(env.action_space, dict):
                    self.action_spaces[agent] = env.action_space[agent]

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def reset(self, seed=None, options=None):
        return self.env.reset(seed=seed, options=options)

    def step(self, actions):
        # RLlib이 일부 에이전트의 액션만 보낼 경우 대비 (KeyError 방지)
        full_actions = actions.copy()
        target_agents = getattr(self.env, "possible_agents", self.env.agents)
        for agent in target_agents:
            if agent not in full_actions:
                full_actions[agent] = 0  # NOOP
        return self.env.step(full_actions)

    def render(self, *args, **kwargs):
        return self.env.render()

class FixedParallelPettingZooEnv(ParallelPettingZooEnv):
    """RLlib의 PettingZooEnv에서 render 인자 문제를 해결한 클래스"""
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
    
    # [수정] render 호출 시 인자가 들어와도 무시하고 내부 env의 render 호출
    def render(self, *args, **kwargs):
        return self.par_env.render()

def env_creator(config=None):
    if config is None: config = {}
    
    # 1. Substrate 로드 (오타 수정됨: common -> commons)
    substrate_name = config.get("substrate", "commons_harvest__partnership") 
    substrate = load_meltingpot(substrate_name)
    
    # 2. Shimmy 변환
    env = MeltingPotCompatibilityV0(substrate, render_mode="rgb_array")
    
    # 3. [핵심] RGB만 남기기 (Dict -> Box 변환)
    # 이 래퍼가 없으면 에러가 발생합니다.
    env = MeltingPotRGBWrapper(env)

    # 4. 프레임 스택 (학습 때 사용했다면 주석 해제 필요)
    # 만약 학습 시 CNN+LSTM이 아니라 FrameStack을 썼다면 아래 주석을 풀어주세요.
    # env = FrameStackWrapper(env, num_stack=4)
    
    # 5. 호환성 래퍼 (공간 정의 및 키 에러 방지)
    env = ShimmyCompatibilityWrapper(env)
        
    # 6. RLlib 호환성 래퍼 (렌더링 픽스)
    env = FixedParallelPettingZooEnv(env)
    
    return env