import os
import ray
import gymnasium as gym
import numpy as np
import imageio
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env
from pettingzoo.utils.wrappers import BaseParallelWrapper
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from shimmy import MeltingPotCompatibilityV0
from shimmy.utils.meltingpot import load_meltingpot

# 사용자 모델 임포트 (이건 파일이 필요합니다)
try:
    from model import MeltingPotModel
except ImportError:
    print("❌ 'model.py' 파일을 찾을 수 없습니다. 현재 폴더에 model.py가 있는지 확인해주세요.")
    exit()

# ==============================================================================
# [설정] 경로 및 파라미터 (여기를 본인 환경에 맞게 수정하세요)
# ==============================================================================
CHECKPOINT_PATH = "/home/jsr/project/Cooperative_pong_RL_agent/Meltingpot_muliagent_env/common_harvest_partnership/results/MeltingPot_Partnership_PPO_Final/PPO_meltingpot_partnership_complete_59211_00000_0_2026-02-04_12-48-20/checkpoint_000054"  # 체크포인트 경로 입력
OUTPUT_VIDEO_PATH = "partnership_result.mp4"
MAX_STEPS = 1000
FPS = 10
SUBSTRATE_NAME = "commons_harvest__partnership" # (오타 수정됨)

# ==============================================================================
# [핵심] 환경 래퍼 클래스 직접 정의 (env_utils.py 의존성 제거)
# ==============================================================================
class DirectRGBWrapper(BaseParallelWrapper):
    """관측값 딕셔너리에서 'RGB'만 강제로 추출하는 래퍼"""
    def __init__(self, env):
        super().__init__(env)
        self.observation_spaces = {}
        agents = getattr(env, "possible_agents", getattr(env, "agents", []))
        
        for agent in agents:
            obs_space = env.observation_space(agent)
            # Dict 공간이면 RGB만 추출해서 Box 공간으로 변경
            if isinstance(obs_space, gym.spaces.Dict):
                self.observation_spaces[agent] = obs_space["RGB"]
            else:
                self.observation_spaces[agent] = obs_space

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
            # 데이터가 딕셔너리고 'RGB' 키가 있으면 그것만 꺼냄
            if isinstance(data, dict) and "RGB" in data:
                new_obs[agent] = data["RGB"]
            else:
                new_obs[agent] = data
        return new_obs
    
    def render(self, *args, **kwargs):
        return self.env.render()

class RLlibCompatWrapper(ParallelPettingZooEnv):
    """RLlib 호환성 및 렌더링 인자 문제 해결"""
    def __init__(self, env):
        super().__init__(env)
    
    def render(self, *args, **kwargs):
        return self.par_env.render()

# ==============================================================================
# [함수] 환경 생성 (직접 정의한 래퍼 사용)
# ==============================================================================
def local_env_creator(config=None):
    # 1. 로드
    substrate = load_meltingpot(SUBSTRATE_NAME)
    # 2. Shimmy 변환
    env = MeltingPotCompatibilityV0(substrate, render_mode="rgb_array")
    # 3. [중요] RGB 추출 (여기서 Dict -> Array 변환됨)
    env = DirectRGBWrapper(env)
    # 4. RLlib 호환
    env = RLlibCompatWrapper(env)
    return env

# ==============================================================================
# [함수] 정책 매핑
# ==============================================================================
def policy_mapping_fn(agent_id, *args, **kwargs):
    if agent_id in ["player_0", "player_1"]:
        return "shared_policy"
    else:
        return "background_policy"

# ==============================================================================
# [메인] 실행 로직
# ==============================================================================
def main():
    if ray.is_initialized():
        ray.shutdown()
    ray.init(ignore_reinit_error=True)

    # 모델 등록
    ModelCatalog.register_custom_model("meltingpot_model", MeltingPotModel)
    
    # 환경 등록 (위에서 만든 함수 사용)
    register_env("meltingpot_partnership_complete", lambda cfg: local_env_creator(cfg))

    print(f"🔄 Loading Checkpoint from: {CHECKPOINT_PATH}")
    
    try:
        algo = Algorithm.from_checkpoint(CHECKPOINT_PATH)
    except Exception as e:
        print(f"❌ Checkpoint Load Error: {e}")
        return

    print("✅ Model Loaded!")

    # 렌더링용 환경 생성
    env = local_env_creator()
    
    obs, info = env.reset()
    
    # 디버깅: 관측값 형태 확인
    first_agent = list(obs.keys())[0]
    print(f"🔎 First observation type: {type(obs[first_agent])}")
    if isinstance(obs[first_agent], dict):
        print("❌ Still getting a Dict! Wrapper failed.")
        print(f"   Keys: {obs[first_agent].keys()}")
        return
    else:
        print(f"✅ Observation is Array! Shape: {obs[first_agent].shape}")

    # LSTM 상태 초기화
    agent_states = {}
    shared_init = algo.get_policy("shared_policy").get_initial_state()
    bg_init = algo.get_policy("background_policy").get_initial_state()
    
    for agent_id in env.par_env.possible_agents:
        pid = policy_mapping_fn(agent_id)
        agent_states[agent_id] = shared_init if pid == "shared_policy" else bg_init

    print("🎬 Start Recording...")
    frames = []
    
    for step in range(MAX_STEPS):
        try:
            frame = env.render()
            if frame is not None:
                frames.append(frame)
        except Exception as e:
            print(f"Render Error: {e}")

        actions = {}
        for agent_id, agent_obs in obs.items():
            pid = policy_mapping_fn(agent_id)
            # RLlib 추론
            action, next_state, _ = algo.compute_single_action(
                observation=agent_obs,
                state=agent_states[agent_id],
                policy_id=pid,
                explore=False
            )
            actions[agent_id] = action
            agent_states[agent_id] = next_state

        obs, rewards, terms, truncs, infos = env.step(actions)
        
        if terms.get("__all__", False) or truncs.get("__all__", False):
            print(f"Done at step {step}")
            break

    env.close()
    
    if frames:
        print(f"💾 Saving {len(frames)} frames to {OUTPUT_VIDEO_PATH}...")
        imageio.mimsave(OUTPUT_VIDEO_PATH, frames, fps=FPS, macro_block_size=None)
        print("🎉 Success!")
    else:
        print("❌ No frames captured.")

    ray.shutdown()

if __name__ == "__main__":
    main()