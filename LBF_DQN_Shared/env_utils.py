import gymnasium as gym
import numpy as np
import lbforaging  # 환경 등록을 위해 필수
from ray.rllib.env.multi_agent_env import MultiAgentEnv

class LBFToRLLibWrapper(MultiAgentEnv):
    """
    Standard Gym (Tuple-based) -> RLLib MultiAgentEnv (Dict-based) 변환 래퍼
    + Distance-based Reward Shaping 추가 (Move closer to food)
    + Success Rate Metric 추가
    """
    def __init__(self, env_config):
        # 1. 환경 ID 설정
        self.env_id = env_config.get("env_id", "Foraging-8x8-2p-2f-v2")
        
        # Reward Shaping 강도
        self.shaping_weight = env_config.get("shaping_weight", 0.01)

        # 2. 환경 생성
        try:
            self._env = gym.make(self.env_id, render_mode="rgb_array")
        except TypeError:
            print(f"[Warning] {self.env_id} does not support render_mode in init. Fallback to default.")
            self._env = gym.make(self.env_id)
        
        # 3. 래퍼 벗겨내기
        if hasattr(self._env, "unwrapped"):
            self.base_env = self._env.unwrapped
        else:
            self.base_env = self._env
            
        self.num_agents = getattr(self.base_env, "n_agents", 2)
        self._agent_ids = [f"agent_{i}" for i in range(self.num_agents)]
        
        # 4. 관측/행동 공간 정의
        self.observation_space = self._env.observation_space[0]
        self.action_space = self._env.action_space[0]
        
        self.prev_min_dists = [0.0] * self.num_agents
        
        # [Metric] 초기 음식 개수 저장 변수
        self.initial_food_count = 0
        
        super().__init__()

    def _get_min_distances(self):
        dists = []
        players = self.base_env.players
        field = self.base_env.field
        food_locations = np.argwhere(field > 0)

        for player in players:
            if len(food_locations) == 0:
                dists.append(0.0)
                continue
            p_pos = np.array(player.position)
            f_dists = [np.linalg.norm(p_pos - f_pos) for f_pos in food_locations]
            min_dist = min(f_dists) if f_dists else 0.0
            dists.append(min_dist)
        return dists

    def reset(self, *, seed=None, options=None):
        reset_result = self._env.reset(seed=seed, options=options)
        
        if isinstance(reset_result, tuple) and len(reset_result) == 2:
            obs_tuple, info = reset_result
        else:
            obs_tuple = reset_result
            info = {}
            
        # [Reward Shaping] 초기 거리 계산
        self.prev_min_dists = self._get_min_distances()
        
        # [Metric] 초기 음식 개수 계산 및 저장
        # field에서 0보다 큰 값이 음식입니다.
        self.initial_food_count = np.count_nonzero(self.base_env.field > 0)

        obs_dict = {}
        for i, agent_id in enumerate(self._agent_ids):
            obs_dict[agent_id] = obs_tuple[i]
            
        return obs_dict, info

    def step(self, action_dict):
        actions = []
        for agent_id in self._agent_ids:
            actions.append(action_dict.get(agent_id, 0))
            
        step_result = self._env.step(tuple(actions))
        
        if len(step_result) == 5:
            obs_tuple, rewards_list, terminated, truncated, info = step_result
            done = terminated or truncated
        elif len(step_result) == 4:
            obs_tuple, rewards_list, done, info = step_result
        else:
            raise ValueError(f"Environment returned unexpected number of values: {len(step_result)}")
        
        # [Reward Shaping]
        current_dists = self._get_min_distances()
        rewards_list = list(rewards_list)

        for i in range(self.num_agents):
            dist_delta = self.prev_min_dists[i] - current_dists[i]
            shaping_reward = dist_delta * self.shaping_weight
            rewards_list[i] += shaping_reward

        self.prev_min_dists = current_dists
        
        # [Metric] 성공률(Success Rate) 계산
        # 현재 남은 음식 개수 확인
        current_food_count = np.count_nonzero(self.base_env.field > 0)
        
        if self.initial_food_count > 0:
            # (초기 음식 - 현재 음식) / 초기 음식
            eaten_count = self.initial_food_count - current_food_count
            success_rate = eaten_count / self.initial_food_count
        else:
            success_rate = 0.0

        obs = {}
        rewards = {}
        terminations = {}
        truncations = {}
        infos = {}
        
        for i, agent_id in enumerate(self._agent_ids):
            obs[agent_id] = obs_tuple[i]
            rewards[agent_id] = rewards_list[i]
            terminations[agent_id] = done
            truncations[agent_id] = False
            
            # [Metric] info에 success_rate 추가
            infos[agent_id] = {"success_rate": success_rate}
            
        terminations["__all__"] = done
        truncations["__all__"] = False
        
        return obs, rewards, terminations, truncations, infos

    def render(self):
        return self._env.render()

    def close(self):
        self._env.close()

def env_creator(config):
    return LBFToRLLibWrapper(config)