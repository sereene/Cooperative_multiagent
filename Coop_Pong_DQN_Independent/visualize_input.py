import numpy as np
import matplotlib.pyplot as plt
import supersuit as ss
from pettingzoo.butterfly import cooperative_pong_v5
from gymnasium import spaces
from collections import deque
from pettingzoo.utils.wrappers import BaseParallelWrapper
from collections import deque

class FrameStackWrapper(BaseParallelWrapper):
    def __init__(self, env, num_stack=4):
        super().__init__(env)
        self.num_stack = num_stack
        
        self.frames = {agent: deque(maxlen=num_stack) for agent in env.possible_agents}
        
        self.h = 84
        self.w = 168
        self.c = 1 # 흑백 이미지 가정
        self.new_channels = self.c * num_stack # 4
        
        # Space 정의
        self.target_space = spaces.Box(
            low=0, 
            high=255, 
            shape=(self.h, self.w, self.new_channels), 
            dtype=np.uint8
        )
        
        # PettingZoo 호환성 업데이트
        for agent in env.possible_agents:
            self.observation_spaces[agent] = self.target_space

    def observation_space(self, agent):
        return self.target_space

    def _process_obs(self, obs):
        """
        입력 관측값을 무조건 (84, 168, 1)로 변환
        """
        # 1. 2D (84, 168) -> (84, 168, 1)
        if obs.ndim == 2:
            return obs[..., None]
        
        # 2. 3D인데 차원이 이상한 경우
        # (1, 84, 168) -> (84, 168, 1)
        if obs.shape == (1, 84, 168):
            return np.transpose(obs, (1, 2, 0))
            
        # 3. 이미 (84, 168, 1)이면 그대로 반환
        return obs

    def _get_stacked_obs(self, agent, obs):
        # 1. 단일 프레임 정규화
        obs = self._process_obs(obs)
        
        # 2. 큐 업데이트
        self.frames[agent].append(obs)
        while len(self.frames[agent]) < self.num_stack:
            self.frames[agent].append(obs)
        
        # # 3. Fading 효과 적용
        # faded_frames = []
        # for i, frame in enumerate(self.frames[agent]):
        #     weight = self.weights[i]
        #     if weight == 1.0:
        #         faded_frames.append(frame)
        #     else:
        #         faded_frame = (frame.astype(np.float32) * weight).astype(np.uint8)
        #         faded_frames.append(faded_frame)
        
        # 4. 합치기 (84, 168, 4)
        stacked_obs = np.concatenate(list(self.frames[agent]), axis=-1)
        
        return np.ascontiguousarray(stacked_obs, dtype=np.uint8)

    def reset(self, seed=None, options=None):
        obs, infos = self.env.reset(seed=seed, options=options)
        for agent in self.frames:
            self.frames[agent].clear()
        
        stacked_obs = {}
        for agent, observation in obs.items():
            stacked_obs[agent] = self._get_stacked_obs(agent, observation)
        return stacked_obs, infos

    def step(self, actions):
        obs, rewards, terminations, truncations, infos = self.env.step(actions)
        
        stacked_obs = {}
        for agent, observation in obs.items():
            stacked_obs[agent] = self._get_stacked_obs(agent, observation)
        return stacked_obs, rewards, terminations, truncations, infos

# ==============================================================================
# [2] 시각화 로직
# ==============================================================================
def visualize_stacking():
    print("Initialize Environment...")
    # 1. 환경 생성
    env = cooperative_pong_v5.parallel_env(render_mode="rgb_array")
    
    # 2. 전처리 (Resize -> GrayScale -> FadingStack)
    env = ss.resize_v1(env, x_size=168, y_size=84)
    env = ss.color_reduction_v0(env, mode="full")
    env = FrameStackWrapper(env, num_stack=4)

    obs, _ = env.reset()
    
    print("Running Simulation loop...")
    # 3. 움직임이 보일 때까지 몇 스텝 실행
    for _ in range(30):
        actions = {agent: env.action_space(agent).sample() for agent in env.possible_agents}
        obs, _, _, _, _ = env.step(actions)

    # 4. 'paddle_0' 에이전트의 관측값 가져오기
    target_agent = env.possible_agents[0]
    agent_obs = obs[target_agent]
    
    print(f"Agent: {target_agent}")
    print(f"Observation Shape: {agent_obs.shape}") # (84, 168, 4) 여야 함

    # 5. 채널별로 분리하여 시각화
    # agent_obs[:, :, 0] -> 가장 오래된 프레임 (가장 어두움)
    # agent_obs[:, :, 3] -> 가장 최신 프레임 (가장 밝음)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f"Frame Stacking Visualization (Shape: {agent_obs.shape})", fontsize=16)

    titles = [
        "Channel 0 (Oldest / T-3)\nDarkest",
        "Channel 1 (T-2)",
        "Channel 2 (T-1)",
        "Channel 3 (Newest / T)\nBrightest"
    ]

    for i, ax in enumerate(axes.flat):
        if i < 4:
            # i번째 채널 추출
            channel_img = agent_obs[:, :, i]
            
            ax.imshow(channel_img, cmap='gray', vmin=0, vmax=255)
            ax.set_title(titles[i])
            ax.axis('off')

    plt.tight_layout()
    plt.savefig("debug_stacking.png")
    print("\n[Complete] Saved visualization to 'debug_stacking.png'")
    
    # 6. (선택) 합쳐서 잔상 확인하기
    plt.figure(figsize=(8, 4))
    combined = np.max(agent_obs, axis=2) # 4장을 겹쳐서 가장 밝은 픽셀 표시
    plt.imshow(combined, cmap='gray', vmin=0, vmax=255)
    plt.title("Combined View (Motion Trail Effect)")
    plt.axis('off')
    plt.savefig("debug_stacking_combined.png")
    print("[Complete] Saved combined view to 'debug_stacking_combined.png'")

if __name__ == "__main__":
    visualize_stacking()