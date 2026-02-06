import numpy as np
import tensorflow as tf
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.view_requirement import ViewRequirement
from ray.rllib.policy.sample_batch import SampleBatch

class TFSavedModelPolicy(Policy):
    """
    [Final Fix v3] Melting Pot SavedModel Policy
    - compute_actions 반환 시 actions를 np.array로 변환하여 unbatch 에러 해결
    - RLLib의 state_batches 구조([Component][Batch])를 정확히 처리
    - 초기 상태 로딩 에러 방지를 위해 Zero State 강제 사용
    """
    def __init__(self, observation_space, action_space, config):
        super().__init__(observation_space, action_space, config)
        self.model_path = config["model_path"]
        
        # GPU 충돌 방지를 위해 봇은 CPU 실행 강제
        with tf.device("/cpu:0"):
            print(f"Loading Raw SavedModel from {self.model_path}...")
            self.saved_model = tf.saved_model.load(self.model_path)
            
            if hasattr(self.saved_model, "step"):
                self.step_fn = self.saved_model.step
            else:
                raise ValueError(f"Model at {self.model_path} has no 'step' method!")

        # View Requirements 설정
        self.view_requirements[SampleBatch.OBS] = ViewRequirement(space=observation_space)
        for i in range(2): # h, c 두 개의 상태
            self.view_requirements[f"state_in_{i}"] = ViewRequirement(
                f"state_out_{i}", shift=-1, space=None, used_for_compute_actions=True, used_for_training=False
            )
            self.view_requirements[f"state_out_{i}"] = ViewRequirement(
                space=None, used_for_compute_actions=True, used_for_training=False
            )
        self.is_recurrent = lambda: True

    def get_initial_state(self):
        # DeepMind R2D2 표준: 128 사이즈의 0벡터 2개 [h, c]
        return [
            np.zeros((128,), dtype=np.float32), 
            np.zeros((128,), dtype=np.float32)
        ]

    def compute_actions(self, obs_batch, state_batches=None, prev_action_batch=None, prev_reward_batch=None, **kwargs):
        batch_size = len(obs_batch)

        # 1. 상태 초기화
        if state_batches is None or len(state_batches) < 2:
            single_state = self.get_initial_state()
            state_batches = [
                np.stack([single_state[0]] * batch_size),
                np.stack([single_state[1]] * batch_size)
            ]

        actions = []
        next_h_list = []
        next_c_list = []

        # 2. 추론 루프
        for i, obs in enumerate(obs_batch):

            # # [수정] 봇을 위해 마지막 3채널(현재 프레임)만 추출
            # # RLLib FrameStack은 채널을 뒤에 붙임 (Old -> New)
            obs_last_frame = obs[:, :, -3:]

            # (A) 입력 준비
            rgb_input = tf.convert_to_tensor(obs_last_frame, dtype=tf.uint8)
            obs_input = {'RGB': rgb_input, 'WORLD.RGB': rgb_input}
            
            step_type = tf.constant(1, dtype=tf.int32)
            reward = tf.constant(0.0, dtype=tf.float32)
            discount = tf.constant(1.0, dtype=tf.float32)

            # (B) 상태 준비
            try:
                h_val = state_batches[0][i]
                c_val = state_batches[1][i]
            except Exception:
                dummy = self.get_initial_state()
                h_val, c_val = dummy[0], dummy[1]

            current_state_tuple = (
                tf.convert_to_tensor(h_val),
                tf.convert_to_tensor(c_val)
            )

            # (C) 모델 실행
            try:
                outputs = self.step_fn(step_type, reward, discount, obs_input, current_state_tuple)
                
                action = outputs[0].numpy()
                next_state_tuple = outputs[1]
                
                actions.append(action)
                next_h_list.append(next_state_tuple[0].numpy())
                next_c_list.append(next_state_tuple[1].numpy())

            except Exception:
                actions.append(self.action_space.sample())
                next_h_list.append(h_val)
                next_c_list.append(c_val)

        # 3. 결과 반환
        # [핵심 수정] actions 리스트를 np.array로 감싸서 반환해야 unbatch 에러가 안 남!
        return np.array(actions), [np.array(next_h_list), np.array(next_c_list)], {}

    def get_weights(self):
        return {}
    def set_weights(self, weights):
        pass