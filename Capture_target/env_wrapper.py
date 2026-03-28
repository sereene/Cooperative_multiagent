import gymnasium as gym
import numpy as np
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from environment import CaptureTarget  # 작성하신 environment.py 파일 임포트

class CaptureTargetMultiAgent(MultiAgentEnv):
    def __init__(self, config=None):
        super().__init__()
        config = config or {}
        
        self.n_agent = config.get("n_agent", 2)
        self.n_target = config.get("n_target", 1)
        self.grid_dim = config.get("grid_dim", (8, 8))
        self.terminate_step = config.get("terminate_step", 60)
        
        # CaptureTarget 환경 초기화
        self.env = CaptureTarget(
            n_target=self.n_target,
            n_agent=self.n_agent,
            grid_dim=self.grid_dim,
            terminate_step=self.terminate_step,
            obs_one_hot=False  # 기본 관측값 사용 (QMIX에서는 Continuous Vector 형태가 유리)
        )
        
        # 에이전트 이름 설정
        self.agents = [f"agent_{i}" for i in range(self.n_agent)]
        self._agent_ids = set(self.agents)
        
        # 개별 에이전트의 관측/행동 공간 정의
        # env_wrapper에서 Dictionary 형태로 설정해주면 train.py에서 Tuple/Group으로 변환됨
        single_obs_shape = (self.env.obs_size[0],)
        self.single_obs_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=single_obs_shape, dtype=np.float32)
        self.single_act_space = gym.spaces.Discrete(self.env.n_action[0])
        
        self.observation_space = gym.spaces.Dict({
            agent: self.single_obs_space for agent in self.agents
        })
        self.action_space = gym.spaces.Dict({
            agent: self.single_act_space for agent in self.agents
        })

    def reset(self, seed=None, options=None):
        obs_array, _ = self.env.reset(seed=seed)
        
        # Array (n_agent, obs_size) -> Dict 변환
        obs_dict = {
            self.agents[i]: obs_array[i].astype(np.float32) for i in range(self.n_agent)
        }
        return obs_dict, {}

    def step(self, action_dict):
        # 환경에 보낼 action 리스트 만들기 (인덱스 순서 보장)
        actions = [action_dict[self.agents[i]] for i in range(self.n_agent)]
        
        obs_array, rewards, done, _, infos = self.env.step(actions)
        
        # 결과값 딕셔너리로 변환
        obs_dict = {self.agents[i]: obs_array[i].astype(np.float32) for i in range(self.n_agent)}
        rew_dict = {self.agents[i]: float(rewards[i]) for i in range(self.n_agent)}
        
        done_dict = {self.agents[i]: bool(done) for i in range(self.n_agent)}
        done_dict["__all__"] = bool(done)  # 모든 에이전트 종료 여부
        
        trunc_dict = {self.agents[i]: False for i in range(self.n_agent)}
        trunc_dict["__all__"] = False
        
        info_dict = {self.agents[i]: {} for i in range(self.n_agent)}
        
        return obs_dict, rew_dict, done_dict, trunc_dict, info_dict

    def render(self, mode="rgb_array"):
        # 환경에서 RGB 프레임 가져오기 (비디오 저장을 위해 필수)
        frame = self.env.render(mode=mode)
        return frame

    def close(self):
        if hasattr(self.env, "viewer") and self.env.viewer is not None:
            self.env.viewer.close()