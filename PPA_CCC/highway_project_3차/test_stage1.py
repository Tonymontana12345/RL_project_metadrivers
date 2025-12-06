"""
Stage 1 모델을 Stage 1 환경에서 간단히 테스트
"""

from sb3_contrib import RecurrentPPO
from config_ppo_curriculum_v2 import CURRICULUM_STAGES, BASE_ENV_CONFIG
from envs.metadrive_env import make_env
import numpy as np

# Stage 1 환경 설정
stage1_config = CURRICULUM_STAGES["stage_1"]
env_config = BASE_ENV_CONFIG.copy()
env_config.update({
    "map": stage1_config["map"],  # S (직선)
    "traffic_density": stage1_config["traffic_density"],  # 0.0
    "random_traffic": stage1_config["random_traffic"],  # False
    "decision_repeat": stage1_config["decision_repeat"],  # 1
    "horizon": stage1_config["horizon"],  # 600
    "num_scenarios": 1,
})

# Stage 1 보상 체계 적용
env_config["use_lateral_reward"] = True
env_config["out_of_road_penalty"] = 5.0
env_config["crash_vehicle_penalty"] = 5.0
env_config["crash_object_penalty"] = 4.0
env_config["crash_sidewalk_penalty"] = 4.0
env_config["driving_reward"] = 2.0
env_config["speed_reward"] = 0.3

print("="*70)
print("🧪 Stage 1 모델을 Stage 1 환경에서 테스트")
print("="*70)
print(f"맵: {env_config['map']} (직선)")
print(f"교통: {env_config['traffic_density']} (없음)")
print(f"horizon: {env_config['horizon']}")
print("="*70 + "\n")

# 모델 로드
try:
    model = RecurrentPPO.load("models/ppo_8_stage_curriculum_v1_stage_1.zip")
    print("✅ Stage 1 모델 로드 완료\n")
except Exception as e:
    print(f"❌ 모델 로드 실패: {e}")
    exit(1)

# 환경 생성
env = make_env(seed=1000, render=False, config=env_config)()

# 10 에피소드 테스트
print("🎮 10 에피소드 테스트 시작...\n")
success_count = 0
rewards = []
lengths = []

for ep in range(10):
    obs, info = env.reset()
    done = False
    total_reward = 0
    steps = 0
    
    lstm_states = None
    episode_start = np.ones((1,), dtype=bool)
    
    while not done:
        action, lstm_states = model.predict(
            obs, 
            state=lstm_states,
            episode_start=episode_start,
            deterministic=True
        )
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        steps += 1
        episode_start = np.zeros((1,), dtype=bool)
    
    rewards.append(total_reward)
    lengths.append(steps)
    
    if info.get("arrive_dest", False):
        success_count += 1
        status = "✅ 성공"
    elif steps >= env_config["horizon"] - 1:
        status = "⏱️  시간초과"
    elif info.get("out_of_road", False):
        status = "🚧 도로이탈"
    elif info.get("crash", False):
        status = "💥 충돌"
    else:
        status = "❓ 기타"
    
    print(f"에피소드 {ep+1:2d}: {status} | 보상: {total_reward:6.1f} | 길이: {steps:3d}")

env.close()

print("\n" + "="*70)
print("📊 결과 요약")
print("="*70)
print(f"평균 보상: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
print(f"평균 길이: {np.mean(lengths):.1f} ± {np.std(lengths):.1f}")
print(f"성공률: {success_count/10*100:.0f}% ({success_count}/10)")
print("="*70)

if success_count >= 5:
    print("\n✅ Stage 1 학습 성공! (성공률 50% 이상)")
elif success_count >= 3:
    print("\n⚠️  Stage 1 학습 부분 성공 (성공률 30-50%)")
else:
    print("\n❌ Stage 1 학습 실패 (성공률 30% 미만)")
    print("💡 직선 도로에서도 실패하는 것은 심각한 문제입니다.")
    print("   보상 체계나 하이퍼파라미터를 재검토해야 합니다.")
