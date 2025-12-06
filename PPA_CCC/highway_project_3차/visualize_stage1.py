"""
Stage 1 모델 시각화 - 실제로 어떻게 운전하는지 확인
"""

from sb3_contrib import RecurrentPPO
from config_ppo_curriculum_v2 import CURRICULUM_STAGES, BASE_ENV_CONFIG
from envs.metadrive_env import make_env
import numpy as np
import time

# Stage 1 환경 설정
stage1_config = CURRICULUM_STAGES["stage_1"]
env_config = BASE_ENV_CONFIG.copy()
env_config.update({
    "map": stage1_config["map"],  # S (직선)
    "traffic_density": stage1_config["traffic_density"],  # 0.0
    "random_traffic": stage1_config["random_traffic"],  # False
    "decision_repeat": stage1_config["decision_repeat"],  # 1
    "horizon": stage1_config["horizon"],  # 1000
    "num_scenarios": 1,
    "use_render": True,  # 렌더링 활성화!
})

# Stage 1 보상 체계 적용
env_config["use_lateral_reward"] = True
env_config["out_of_road_penalty"] = 2.0
env_config["crash_vehicle_penalty"] = 3.0
env_config["crash_object_penalty"] = 2.0
env_config["crash_sidewalk_penalty"] = 2.0
env_config["driving_reward"] = 4.0
env_config["speed_reward"] = 1.0

print("="*70)
print("🎬 Stage 1 모델 시각화 (직선 도로)")
print("="*70)
print("창이 열리면 에이전트의 주행을 관찰하세요.")
print("ESC 키를 누르면 종료됩니다.")
print("="*70 + "\n")

# 모델 로드
try:
    model = RecurrentPPO.load("models/ppo_8_stage_curriculum_v1_stage_1.zip")
    print("✅ Stage 1 모델 로드 완료\n")
except Exception as e:
    print(f"❌ 모델 로드 실패: {e}")
    print("최종 모델을 사용합니다...")
    model = RecurrentPPO.load("models/ppo_8_stage_curriculum_v1_final.zip")

# 환경 생성
env = make_env(seed=1000, render=True, config=env_config)()

# 3 에피소드 시각화
for ep in range(3):
    print(f"\n🎮 에피소드 {ep+1}/3 시작...")
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
        
        # 천천히 렌더링 (30 FPS)
        time.sleep(1/30)
    
    # 결과 출력
    if info.get("arrive_dest", False):
        status = "✅ 성공"
    elif steps >= env_config["horizon"] - 1:
        status = "⏱️  시간초과"
    elif info.get("out_of_road", False):
        status = "🚧 도로이탈"
    elif info.get("crash", False):
        status = "💥 충돌"
    else:
        status = "❓ 기타"
    
    print(f"에피소드 {ep+1}: {status} | 보상: {total_reward:.1f} | 길이: {steps}")
    print(f"행동: steering={action[0]:.3f}, throttle={action[1]:.3f}")
    
    if ep < 2:
        print("다음 에피소드를 위해 3초 대기...")
        time.sleep(3)

env.close()
print("\n✅ 시각화 완료")
