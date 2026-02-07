import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym
from CNN_LSTM_model import MeltingPotModel

def test_meltingpot_lstm():
    print("="*60)
    print("🧪 [Test] MeltingPot LSTM Model Integrity Check")
    print("="*60)

    # ---------------------------------------------------------
    # 1. 환경 및 모델 설정 (Mocking)
    # ---------------------------------------------------------
    # 가짜 관측 공간 (88x88 RGB 이미지)
    obs_space = gym.spaces.Box(0, 255, shape=(88, 88, 3), dtype=np.uint8)
    
    # 가짜 행동 공간 (Discrete 8)
    action_space = gym.spaces.Discrete(8)
    
    # 모델 초기화
    model_config = {"custom_model_config": {}}
    model = MeltingPotModel(
        obs_space=obs_space,
        action_space=action_space,
        num_outputs=8,
        model_config=model_config,
        name="test_model"
    )
    
    print(f"✅ Model Initialized: {model}")

    # ---------------------------------------------------------
    # 2. 입력 데이터 생성 (Batch=2, Time=5)
    # ---------------------------------------------------------
    B, T = 2, 5
    input_dict = {
        "obs": torch.randint(0, 255, (B, T, 88, 88, 3), dtype=torch.float32), # 이미지
        "prev_actions": torch.randint(0, 8, (B, T)).long(),
        "prev_rewards": torch.randn(B, T)
    }
    
    # 초기 상태 가져오기 (h, c)
    state = model.get_initial_state()
    # Batch 크기에 맞게 상태 확장 (RLLib 내부 동작 모방)
    # state는 [Hidden_State, Cell_State] 리스트
    # 각 텐서는 [Batch, Hidden_Size] 여야 함
    state = [s.unsqueeze(0).repeat(B, 1) for s in state] 
    
    seq_lens = torch.LongTensor([T] * B) # 모든 시퀀스 길이는 5

    print(f"\n📊 Input Shape: {input_dict['obs'].shape} (Batch={B}, Time={T})")
    print(f"📊 Initial State Shape: h={state[0].shape}, c={state[1].shape}")

    # ---------------------------------------------------------
    # 3. Forward Pass (전파)
    # ---------------------------------------------------------
    output, new_state = model(input_dict, state, seq_lens)
    
    # 검증 1: 출력 크기 확인
    expected_shape = (B * T, 8) # [10, 8]
    assert output.shape == expected_shape, f"❌ Output shape mismatch! Expected {expected_shape}, got {output.shape}"
    print(f"✅ Forward Pass Successful. Output Shape: {output.shape}")

    # 검증 2: 상태 업데이트 확인 (기억이 변했는가?)
    # 초기 상태(0)와 새 상태가 달라야 함
    is_state_updated = not torch.allclose(state[0], new_state[0])
    if is_state_updated:
        print("✅ LSTM State Updated: Memory is changing based on input.")
    else:
        print("❌ Warning: LSTM State did NOT change. (Check if inputs are all zero or gradients are disconnected)")

    # ---------------------------------------------------------
    # 4. Backward Pass (학습 신호 전달)
    # ---------------------------------------------------------
    # 임의의 손실 함수 계산 (Mean Squared Error)
    target = torch.randn_like(output)
    loss = nn.functional.mse_loss(output, target)
    
    # 역전파
    loss.backward()
    
    # LSTM 가중치에 Gradient가 맺혔는지 확인
    lstm_weight_grad = model.lstm.weight_ih_l0.grad
    if lstm_weight_grad is not None and torch.sum(torch.abs(lstm_weight_grad)) > 0:
        grad_norm = torch.norm(lstm_weight_grad).item()
        print(f"✅ Gradient Flow Confirmed! (LSTM Weight Grad Norm: {grad_norm:.4f})")
    else:
        print("❌ Error: No gradient flow to LSTM. The model will not learn.")

    
    print("\n🎉 Test Complete. If all checks passed, your LSTM implementation is correct.")

if __name__ == "__main__":
    try:
        test_meltingpot_lstm()
    except Exception as e:
        print(f"\n❌ Test Failed with Exception: {e}")
        import traceback
        traceback.print_exc()
