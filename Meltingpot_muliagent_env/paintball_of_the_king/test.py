import tensorflow as tf

MODEL_PATH = "/home/jsr/project/Cooperative_pong_RL_agent/Meltingpot_muliagent_env/meltingpot_repo/assets/saved_models/paintball__king_of_the_hill/free_bot_0"

print(f"Loading model from {MODEL_PATH}...")
loaded = tf.saved_model.load(MODEL_PATH)

print("\n=== Loaded Object Attributes (숨겨진 함수 찾기) ===")
# 객체가 가진 모든 속성/함수 이름 출력
attributes = dir(loaded)
print(attributes)

print("\n=== Critical Methods Check ===")
if 'step' in attributes:
    print("✅ Found 'step' method! (이걸 쓰면 됩니다)")
if 'initial_state' in attributes:
    print("✅ Found 'initial_state' method!")
if '__call__' in attributes:
    print("✅ Object is callable (함수처럼 호출 가능)")