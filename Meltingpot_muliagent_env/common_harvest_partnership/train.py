import os
import torch
import ray
from ray import tune
from ray.tune.registry import register_env
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.models import ModelCatalog
from ray.rllib.policy.policy import Policy
from ray.air.integrations.wandb import WandbLoggerCallback
from datetime import datetime

# 작성한 모듈 임포트
from model import MeltingPotModel
from env_utils import env_creator
from callbacks import GifCallbacks

# [설정] 불러올 모델 경로 (Agents 2~6용 - 침입자)
BACKGROUND_MODEL_PATH = "/home/jsr/project/Cooperative_pong_RL_agent/Meltingpot_muliagent_env/meltingpot_repo/assets/saved_models/commons_harvest__partnership/free_2"

# 가중치 로드 유틸리티
def load_background_weights(algorithm, policy_id, path):
    print(f"Loading weights for '{policy_id}' from {path}...")
    try:
        if os.path.isdir(path):
            checkpoint_policy = Policy.from_checkpoint(path)
            weights = checkpoint_policy.get_weights()
        else:
            weights = torch.load(path)
        
        algorithm.set_weights({policy_id: weights})
        print(f"Successfully loaded weights for '{policy_id}'")
    except Exception as e:
        print(f"[Warning] Failed to load background model: {e}")
        print("Proceeding with random weights for background agents.")

if __name__ == "__main__":
    ray.init()
    
    # 커스텀 모델 등록
    ModelCatalog.register_custom_model("meltingpot_model", MeltingPotModel)
    
    env_name = "meltingpot_partnership_complete"
    register_env(env_name, lambda cfg: env_creator(cfg))

    # [수정됨] 환경 스펙 확인
    tmp_env = env_creator()
    
    # RLlib 래퍼는 observation_space를 단일 에이전트의 것으로 바로 노출합니다.
    # 따라서 ["player_0"] 인덱싱을 제거해야 합니다.
    obs_space = tmp_env.observation_space
    act_space = tmp_env.action_space
    
    # 에이전트 목록은 내부 PettingZoo 환경에서 가져옵니다.
    possible_agents = tmp_env.par_env.possible_agents
    print(f"DEBUG: Agents List: {possible_agents}")
    print(f"DEBUG: Obs Space keys: {list(obs_space.keys()) if hasattr(obs_space, 'keys') else 'Not a Dict'}")
    
    tmp_env.close()

    # [정책 정의]
    policies = {
        "shared_policy": (None, obs_space, act_space, {}),      # 학습 대상 (Agent 0, 1)
        "background_policy": (None, obs_space, act_space, {}),  # 고정 대상 (Agent 2~6)
    }

    def policy_mapping_fn(agent_id, *args, **kwargs):
        # player_0, player_1은 shared_policy
        if agent_id in ["player_0", "player_1"]:
            return "shared_policy"
        # 나머지는 background_policy
        else:
            return "background_policy"

    current_dir = os.getcwd()
    local_log_dir = os.path.join(current_dir, "results")
    exp_name = "MeltingPot_Partnership_PPO_Final"
    start_time = datetime.now().strftime("%m-%d_%H-%M-%S")
    gif_save_path = os.path.join(local_log_dir, exp_name, f"gifs_{start_time}")

    # [PPO Config]
    config = (
        PPOConfig()
        .rl_module(_enable_rl_module_api=False)
        .training(_enable_learner_api=False)

        .environment(env=env_name, clip_actions=True, disable_env_checking=True)

        .framework("torch")
        
        .rollouts(
            num_rollout_workers=8,        # CPU 코어 수에 맞게 조정
            rollout_fragment_length=256,  # [Paper] Unroll length
        )
        
        .training(
            model={
                "custom_model": "meltingpot_model",
                "max_seq_len": 100,       
                "vf_share_layers": False, 
            },
            
            # [Paper Hyperparameters]
            lr=1e-5,
            entropy_coeff=0.00,
            sgd_minibatch_size=256,
            train_batch_size=256*8, 
            gamma=0.99,
            lambda_=1.0,           
            vf_loss_coeff=0.5,
            clip_param=0.2,
            grad_clip=10.0,        
        )
        
        .multi_agent(
            policies=policies,
            policy_mapping_fn=policy_mapping_fn,
            policies_to_train=["shared_policy"],
        )
        
        # GIF 콜백 연결
        .callbacks(lambda: GifCallbacks(out_dir=gif_save_path))
        
        .evaluation(
            evaluation_interval=20,
            evaluation_num_episodes=1,
            evaluation_config={"explore": False},
        )
        .resources(num_gpus=1 if torch.cuda.is_available() else 0)
    )
    
    # [Mixin] GifCallbacks에 가중치 로드 기능 주입
    # 이렇게 하면 콜백 클래스 내부를 수정하지 않고도 on_algorithm_init을 추가할 수 있습니다.
    def on_algorithm_init_mixin(self, *, algorithm, **kwargs):
        load_background_weights(algorithm, "background_policy", BACKGROUND_MODEL_PATH)
        
    GifCallbacks.on_algorithm_init = on_algorithm_init_mixin

    print(f"### Starting Training. Logs: {local_log_dir} ###")

    tune.run(
        "PPO",
        name=exp_name,
        stop={"timesteps_total": 20_000_000},
        local_dir=local_log_dir,
        config=config.to_dict(),
        checkpoint_freq=50,
        checkpoint_at_end=True,
        callbacks=[
            WandbLoggerCallback(
                project="MeltingPot_Partnership",
                group="PPO_Fixed",
                job_type="training",
                name=exp_name,
                log_config=True
            )
        ]
    )