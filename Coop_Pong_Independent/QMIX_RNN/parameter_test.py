import os
import torch
import ray
from ray.tune.registry import register_env
from ray.rllib.algorithms.qmix import QMixConfig
import gymnasium as gym
from ray.rllib.models import ModelCatalog

# 질문자님의 기존 모듈들
from models_CNNGRU import CustomCNNGRU
from env_utils import env_creator

def check_qmix_parameters():
    ray.init(ignore_reinit_error=True)

    ModelCatalog.register_custom_model("custom_cnn_gru", CustomCNNGRU)

    # 1. 환경 및 그룹 설정 (기존과 동일)
    grouping = {"group_1": ["paddle_0", "paddle_1"]}
    def grouped_env_creator(config):
        env = env_creator(config)
        return env.with_agent_groups(grouping, obs_space=env.observation_space, act_space=env.action_space)

    env_name = "cooperative_pong_qmix"
    register_env(env_name, grouped_env_creator)

    base_env = env_creator({})
    single_obs_space = base_env.observation_space
    single_act_space = base_env.action_space
    base_env.close()

    group_obs_space = gym.spaces.Tuple([single_obs_space, single_obs_space])
    group_act_space = gym.spaces.Tuple([single_act_space, single_act_space])

    # 2. RLlib QMIX Config 설정 (문제의 QMIX 믹서 설정)
    config = (
        QMixConfig()
        .rl_module(_enable_rl_module_api=False)
        .training(_enable_learner_api=False)
        .environment(env=env_name, disable_env_checking=True)
        .framework("torch")
        .multi_agent(
            policies={"group_1": (None, group_obs_space, group_act_space, {})},
            policy_mapping_fn=lambda agent_id, *args, **kwargs: "group_1"
        )
        .training(
            mixer="qmix",             # VDN이 아닌 QMIX 믹서 활성화!
            mixing_embed_dim=32,      # 기본 임베딩 차원
            model={
                "custom_model": "custom_cnn_gru", # (필요시 ModelCatalog 등록 후 사용)
                "max_seq_len": 20,
            }
        )
    )

    print("\\n[1/3] RLlib 알고리즘(모델)을 빌드 중입니다... (약 5~10초 소요)")
    # 💡 핵심: 학습을 돌리지 않고 알고리즘 객체와 모델만 메모리에 생성합니다.
    algo = config.build()

    print("[2/3] 정책(Policy)과 믹서(Mixer) 모델을 추출합니다...")
    policy = algo.get_policy("group_1")
    mixer = policy.mixer  # model을 거치지 않고 policy에서 직접 mixer를 가져옵니다.

    # 3. 파라미터 측정
    print("\\n=== 🚨 RLlib 내부 QMixer 실제 파라미터 수 측정 결과 ===")
    
    # 전체 믹서 파라미터 계산
    total_params = sum(p.numel() for p in mixer.parameters() if p.requires_grad)
    
    # 믹서 내부 구조 출력 (Hypernetwork 등)
    print(f"\\n▶ RLlib Mixer 아키텍처:\\n{mixer}")
    
    # 첫 번째 선형 계층(Hypernetwork layer 1) 파라미터만 추출해서 확인
    # RLlib의 QMixer는 내부적으로 hyper_w_1 이라는 이름을 사용합니다.
    if hasattr(mixer, 'hyper_w_1'):
        layer1_params = sum(p.numel() for p in mixer.hyper_w_1.parameters())
        print(f"\\n▶ 첫 번째 Hypernetwork 계층 (hyper_w_1) 파라미터 수: {layer1_params:,} 개")
    
    print(f"▶ QMixer '전체' 파라미터 수: {total_params:,} 개")
    print("========================================================\\n")

    ray.shutdown()

if __name__ == "__main__":
    check_qmix_parameters()