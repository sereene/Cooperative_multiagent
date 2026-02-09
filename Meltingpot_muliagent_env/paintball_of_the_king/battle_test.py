import os
import sys
import collections
import cv2
import numpy as np
import ray
import dm_env
import shimmy
import torch

from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env

# [DeepMind Melting Pot Policy Import]
# 경로 설정이 필요하다면 sys.path.append를 사용하세요.
try:
    from meltingpot.utils.policies import saved_model_policy
except ImportError:
    # 예: meltingpot_repo가 현재 폴더에 있다면
    try:
        from meltingpot_repo.meltingpot.utils.policies import saved_model_policy
    except ImportError:
        print("⚠️ [Error] 'meltingpot' 패키지를 찾을 수 없습니다. 경로를 확인해주세요.")
        sys.exit(1)

# [사용자 파일 Import] (env_utils.py, model.py가 같은 폴더에 있어야 함)
try:
    from env_utils import env_creator
    from model import MeltingPotModel
except ImportError:
    print("❌ [Error] 'env_utils.py' 또는 'model.py'를 찾을 수 없습니다.")
    sys.exit(1)

# ==============================================================================
# 1. 설정 (경로 및 파라미터)
# ==============================================================================
# [User] 학습된 체크포인트
USER_CHECKPOINT_PATH = "/home/jsr/project/Cooperative_pong_RL_agent/Meltingpot_muliagent_env/paintball_of_the_king/results_selfplay/MeltingPot_KOTH_SelfPlay_noBot_1e-5_Fc256/PPO_meltingpot_paintball_koth_mixed_70817_00000_0_2026-02-06_20-24-45/checkpoint_000195"

# [Bot] 배경 봇 SavedModel
BOT_MODEL_DIR = "/home/jsr/project/Cooperative_pong_RL_agent/Meltingpot_muliagent_env/meltingpot_repo/assets/saved_models/paintball__king_of_the_hill/free_bot_0"

NUM_EPISODES = 5
RENDER_SCALE = 4     # 화면 확대 배율
FPS = 15             # 영상 저장 및 재생 속도
VIDEO_DIR = "videos" # 영상 저장 폴더

# ==============================================================================
# 2. 에이전트 클래스 정의
# ==============================================================================

class UserAgent:
    """
    RLlib Checkpoint를 사용하는 에이전트
    - 특징: 학습 환경과 동일하게 FrameStack(4)을 수동으로 적용
    """
    def __init__(self, algorithm, policy_id, device):
        self.algo = algorithm
        self.policy_id = policy_id
        self.device = device
        
        # Frame Stack 관리 (학습 설정과 동일하게 4프레임)
        self.num_stack = 3
        self.frames = collections.deque(maxlen=self.num_stack)
        
        # 내부 상태 (LSTM 등 사용 시 필요, 현재 모델은 Stateless지만 호환성 유지)
        self.state = [] 
        
        # 초기화: 정책에서 초기 상태 가져오기
        policy = self.algo.get_policy(self.policy_id)
        if policy:
            self.state = policy.get_initial_state()

    def reset(self, initial_obs):
        """에피소드 시작 시 스택 초기화"""
        self.frames.clear()
        # 초기 프레임으로 스택 채우기
        processed = self._process_obs(initial_obs)
        for _ in range(self.num_stack):
            self.frames.append(processed)
            
        # 상태 초기화
        policy = self.algo.get_policy(self.policy_id)
        self.state = policy.get_initial_state()
    
    def _process_obs(self, obs):
        """입력값 보정 및 RGB 추출"""
        if isinstance(obs, dict) and 'RGB' in obs:
            img = obs['RGB']
        else:
            img = obs
            
        # [핵심] 만약 이미지가 float(0.0~1.0)으로 들어오면 255를 곱해서 복구해줘야 함
        if img.dtype == np.float32 and img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
            
        return img

    def act(self, obs):
        """행동 결정"""
        # 1. 프레임 스택 업데이트
        current_frame = self._process_obs(obs)
        self.frames.append(current_frame)
        
        # 2. 스택 결합 (Channel Concatenation) -> (88, 88, 12)
        # numpy stack은 (4, 88, 88, 3) -> reshape or concatenate -> (88, 88, 12)
        # RLlib FrameStackWrapper의 방식: Concatenate along last axis
        stacked_obs = np.concatenate(list(self.frames), axis=-1)
        
        # 3. 행동 추론 (compute_single_action)
        # full_fetch=True를 해야 state 관리가 용이함
        result = self.algo.compute_single_action(
            observation=stacked_obs,
            state=self.state,
            policy_id=self.policy_id,
            explore=True,  # 평가 모드 (결정론적 행동)
            full_fetch=True
        )
        
        # 4. 결과 언패킹
        if isinstance(result, tuple) and len(result) >= 3:
            action, state_out, _ = result
        else:
            action = result
            state_out = self.state

        self.state = state_out
        return action


class BotAgent:
    """
    Melting Pot 공식 SavedModelPolicy를 사용하는 봇
    """
    def __init__(self, model_path):
        if not os.path.exists(model_path):
            print(f"❌ [BotAgent] 경로 없음: {model_path}")
            sys.exit(1)
            
        self.policy = saved_model_policy.SavedModelPolicy(model_path)
        self.state = self.policy.initial_state()

    def reset(self):
        self.state = self.policy.initial_state()

    def act(self, obs):
        # Shimmy Dict Obs -> dm_env.TimeStep 변환
        timestep = dm_env.TimeStep(
            step_type=dm_env.StepType.MID,
            reward=0.0,
            discount=1.0,
            observation=obs 
        )
        
        # 정책 실행
        action, next_state = self.policy.step(timestep, self.state)
        self.state = next_state
        return int(action)

# ==============================================================================
# 3. 메인 실행 루프
# ==============================================================================
def main():
    # -------------------------------------------------------------------------
    # A. 초기화 (Ray, Model, Env)
    # -------------------------------------------------------------------------
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)

    # Custom Model 등록
    ModelCatalog.register_custom_model("meltingpot_model", MeltingPotModel)
    
    # Env 등록 (Checkpoint 로딩용, 실제 실행은 Raw Shimmy Env 사용)
    env_name = "meltingpot_paintball_koth_mixed"
    register_env(env_name, lambda cfg: env_creator({"substrate": "paintball__king_of_the_hill"}))

    print(f"🔄 Checkpoint Loading: {USER_CHECKPOINT_PATH}")
    try:
        # RLlib 알고리즘 로드
        user_algo = Algorithm.from_checkpoint(USER_CHECKPOINT_PATH)
    except Exception as e:
        print(f"❌ Checkpoint 로드 실패: {e}")
        return

    # 실제 게임 환경 생성 (Shimmy Raw Env - 봇 호환성을 위해 Wrapper 최소화)
    # Bot은 Raw Dict를 원하고, User는 Stacked Array를 원하므로
    # 환경은 Raw로 두고 UserAgent 내부에서 Stack 처리를 합니다.
    env = shimmy.MeltingPotCompatibilityV0(
        substrate_name="paintball__king_of_the_hill",
        render_mode="rgb_array"
    )
    raw_env = env.par_env if hasattr(env, 'par_env') else env
    possible_agents = raw_env.possible_agents
    
    print(f"✅ 환경 생성 완료. Agents: {possible_agents}")
    
    # 영상 저장 폴더 생성
    os.makedirs(VIDEO_DIR, exist_ok=True)

    # -------------------------------------------------------------------------
    # B. 에이전트 인스턴스 생성
    # -------------------------------------------------------------------------
    # Player 0, 2 (User) / Player 1, 3 (Bot)
    agents = {}
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for agent_id in possible_agents:
        # agent_id format: "player_0", "player_1", ...
        idx = int(agent_id.split("_")[-1])
        
        if idx % 2 == 0: # 짝수: User (Red Team)
            print(f" -> {agent_id}: UserAgent (RLlib Main Policy)")
            # 체크포인트 내 정책 이름 매핑 (train.py의 policy_mapping_fn 참조)
            # 보통 0,2는 'main_policy'로 훈련됨
            agents[agent_id] = UserAgent(user_algo, "main_policy", device)
        else: # 홀수: Bot (Blue Team)
            print(f" -> {agent_id}: BotAgent (Official Policy)")
            agents[agent_id] = BotAgent(BOT_MODEL_DIR)

    # -------------------------------------------------------------------------
    # C. 에피소드 루프
    # -------------------------------------------------------------------------
    win_stats = {"User": 0, "Bot": 0, "Draw": 0}

    for ep in range(1, NUM_EPISODES + 1):
        print(f"\n🎬 Episode {ep}/{NUM_EPISODES} Start...")
        
        obs, infos = env.reset()
        done = False
        step_count = 0
        
        # 점수 초기화
        ep_rewards = {aid: 0.0 for aid in possible_agents}
        
        # 에이전트 상태 초기화
        for aid, agent in agents.items():
            if isinstance(agent, UserAgent):
                agent.reset(obs[aid])
            elif isinstance(agent, BotAgent):
                agent.reset()

        # 비디오 설정 (첫 프레임 렌더링 후 크기 결정)
        video_writer = None
        video_path = os.path.join(VIDEO_DIR, f"ep_{ep:03d}_user_vs_bot.mp4")

        try:
            while not done:
                # 1. 행동 결정
                actions = {}
                for agent_id in obs.keys():
                    if agent_id in agents:
                        act = agents[agent_id].act(obs[agent_id])
                        actions[agent_id] = act
                
                # 2. 환경 진행
                next_obs, rewards, terminations, truncations, infos = env.step(actions)
                
                # 3. 리워드 집계
                for aid, r in rewards.items():
                    ep_rewards[aid] += r
                
                # 4. 종료 조건
                if any(terminations.values()) or all(truncations.values()):
                    done = True
                
                obs = next_obs
                step_count += 1

                # 5. 렌더링 및 영상 저장
                frame = env.render() # (H, W, 3) RGB
                if frame is not None:
                    # RGB -> BGR (OpenCV용)
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    
                    # 확대
                    h, w, _ = frame_bgr.shape
                    frame_resized = cv2.resize(frame_bgr, (w * RENDER_SCALE, h * RENDER_SCALE), interpolation=cv2.INTER_NEAREST)
                    
                    # 점수 오버레이
                    user_score = sum(ep_rewards[a] for a in possible_agents if int(a.split("_")[-1]) % 2 == 0)
                    bot_score = sum(ep_rewards[a] for a in possible_agents if int(a.split("_")[-1]) % 2 != 0)
                    
                    cv2.putText(frame_resized, f"Ep {ep} | Step {step_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    cv2.putText(frame_resized, f"User(Red): {user_score:.1f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.putText(frame_resized, f"Bot(Blue): {bot_score:.1f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

                    # 화면 표시
                    cv2.imshow("User vs Bot", frame_resized)
                    if cv2.waitKey(1) & 0xFF == 27: # ESC로 중단 가능
                        print("User interrupted.")
                        env.close()
                        return

                    # 비디오 초기화 (최초 1회)
                    if video_writer is None:
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        video_writer = cv2.VideoWriter(video_path, fourcc, FPS, (frame_resized.shape[1], frame_resized.shape[0]))
                    
                    # 프레임 쓰기
                    video_writer.write(frame_resized)

        finally:
            if video_writer:
                video_writer.release()
                print(f"   💾 Video saved: {video_path}")

        # 결과 집계
        user_total = sum(ep_rewards[a] for a in possible_agents if int(a.split("_")[-1]) % 2 == 0)
        bot_total = sum(ep_rewards[a] for a in possible_agents if int(a.split("_")[-1]) % 2 != 0)
        
        print(f"🏁 Episode {ep} Result: User {user_total:.1f} vs Bot {bot_total:.1f}")
        
        if user_total > bot_total:
            win_stats["User"] += 1
        elif bot_total > user_total:
            win_stats["Bot"] += 1
        else:
            win_stats["Draw"] += 1

    # -------------------------------------------------------------------------
    # D. 종료
    # -------------------------------------------------------------------------
    env.close()
    cv2.destroyAllWindows()
    ray.shutdown()
    
    print("\n" + "="*50)
    print(f"📊 Final Stats (Total {NUM_EPISODES} Games)")
    print(f"   User Wins: {win_stats['User']}")
    print(f"   Bot Wins : {win_stats['Bot']}")
    print(f"   Draws    : {win_stats['Draw']}")
    print("="*50)

if __name__ == "__main__":
    main()