import supersuit as ss
from pettingzoo.butterfly import cooperative_pong_v5
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv

# 기존 Wrapper 파일 import
from Coop_Pong_DQN_Independent.RewardShapingWrapper import RewardShapingWrapper
from Coop_Pong_DQN_Independent.MirrorObservationWrapper import MirrorObservationWrapper

# MAX_CYCLES = 900
MAX_CYCLES = 500

class FixedParallelPettingZooEnv(ParallelPettingZooEnv):
    def __init__(self, env):
        super().__init__(env)
        self.agents = self.par_env.possible_agents
        self._agent_ids = set(self.agents)

    def reset(self, *, seed=None, options=None):
        obs, info = self.par_env.reset(seed=seed, options=options)
        return obs, info

def env_creator(config=None):
    env = cooperative_pong_v5.parallel_env(
        max_cycles=MAX_CYCLES,
        render_mode="rgb_array"
    )
    # env = ss.max_observation_v0(env,2)  # 최근 2프레임 중 픽셀값 최대치 취함

    # 가로 168, 세로 84 크기로 줄이기
    env = ss.resize_v1(env, x_size=84, y_size=168)

    # # 흑백으로 변환하면 메모리를 1/3 더 줄일 수 있음
    # env = ss.color_reduction_v0(env, mode="full")

    # # SuperSuit으로 프레임 4개 강제 스택 (채널이 3 -> 12로 바뀜)
    # env = ss.frame_stack_v1(env, 3)

    # reward shaping wrapper 추가
    env = RewardShapingWrapper(env)

    # Mirror Observation Wrapper 추가
    env = MirrorObservationWrapper(env)

    return FixedParallelPettingZooEnv(env)