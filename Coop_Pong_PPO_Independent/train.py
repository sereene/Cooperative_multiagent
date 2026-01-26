import os
import warnings
import torch
import ray
from ray import tune
from ray.tune.registry import register_env
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.models import ModelCatalog
from ray.air.integrations.wandb import WandbLoggerCallback

# 분리한 모듈 import
from PPO_Independent.models_CNN import CustomCNN
from env_utils import env_creator
from callbacks import GifCallbacks

# 경고 메시지 무시
warnings.filterwarnings("ignore", category=DeprecationWarning)

# 평가 지표 누락 시 에러 무시
os.environ["TUNE_DISABLE_STRICT_METRIC_CHECKING"] = "1"

if __name__ == "__main__":
    ray.init()

    env_name = "cooperative_pong_independent_reward_shaping_CNN"
    register_env(env_name, lambda cfg: env_creator(cfg))
    
    tmp_env = env_creator({})
    obs_space = tmp_env.observation_space
    act_space = tmp_env.action_space
    tmp_env.close()

    # Model 등록
    ModelCatalog.register_custom_model("custom_cnn", CustomCNN)

    # # Shared Policy 정의: 하나의 정책("shared_policy")만 생성
    # policies = {
    #     "shared_policy": (None, obs_space, act_space, {}),
    # }

    # # Policy Mapping 함수: 모든 에이전트 ID를 "shared_policy"로 매핑
    # def policy_mapping_fn(agent_id, *args, **kwargs):
    #     return "shared_policy"


    #각 에이전트 별로 별도 정책 사용
    policies = {
        "policy_left": (None, obs_space, act_space, {}),
        "policy_right": (None, obs_space, act_space, {}),
    }

    def policy_mapping_fn(agent_id, *args, **kwargs):
        return "policy_left" if "0" in agent_id else "policy_right"

    current_dir = os.getcwd()
    local_log_dir = os.path.join(current_dir, "results")

    experiment_name = "PPO_cooperative_pong_newRewardShaping_independent_CNN_5e-6_no_frameStack_mirrorObs"

    gif_save_path = os.path.join(local_log_dir, experiment_name, "gifs")

    config = (
        PPOConfig()
        .rl_module(_enable_rl_module_api=False)
        .training(
            _enable_learner_api=False
        )
        .environment(
            env=env_name, 
            clip_actions=True,
            disable_env_checking=True 
        )
        .framework("torch")
        .rollouts(
            num_rollout_workers=8,
            rollout_fragment_length=256,
            compress_observations=True 
        )
        .training(
            model={
                "custom_model": "custom_cnn",
            },
            train_batch_size=8 * 256,
            sgd_minibatch_size=256,
            num_sgd_iter=8,
            lr=5e-6, # 2e-6
            gamma=0.99,
            lambda_=0.95,
            use_gae=True,
            vf_loss_coeff=0.5,
            entropy_coeff=0.0,
        )
        .callbacks(lambda: GifCallbacks(
          out_dir=gif_save_path,
          every_n_evals=5,
          max_cycles=500,
        ))
        # .callbacks(CoopPongCallbacks)
        .multi_agent(
            policies=policies,
            policy_mapping_fn=policy_mapping_fn,
        )
        .evaluation(
            evaluation_interval=100,
            evaluation_num_episodes=25,
            evaluation_config={
                "explore": False,
            },
        )
        .resources(
            num_gpus=1 if torch.cuda.is_available() else 0
        )
    )

    print(f"### Training Logs will be saved at: {local_log_dir} ###")

    tune.run(
        "PPO",
        name=experiment_name,
        stop={"timesteps_total": 20_000_000},
        local_dir=local_log_dir,
        metric="evaluation/custom_metrics/success_mean",
        mode="max",
        keep_checkpoints_num=2,
        checkpoint_score_attr="evaluation/custom_metrics/success_mean",
        checkpoint_freq=100,
        checkpoint_at_end=True,
        config=config.to_dict(),

        callbacks=[
            WandbLoggerCallback(
                project="cooperative_pong_multiagent_independent",
                group="ppo_experiments",
                job_type="training",
                name=experiment_name,
                log_config=True
            )
        ]
    )