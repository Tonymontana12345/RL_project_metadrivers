"""
PPO 커리큘럼 러닝 전용 설정 파일 (최종 튜닝 버전)

- 5단계 Curriculum
- Stage 3는 TSC (TSCO보다 한 단계 낮은 복잡도)
- Stage 4/5에서만 TSCO 사용
- Stage별 horizon, decision_repeat, num_scenarios 조정
- PPO 설정은 MetaDrive류 연속 제어 환경용 Best Practice 기반
"""

import os

# ============================================================
# 프로젝트 경로
# ============================================================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

# ============================================================
# 기본 환경 및 PPO 설정
# ============================================================

# 모든 단계에 공통적으로 적용될 기본 환경 설정 (stage별로 덮어쓰기됨)
BASE_ENV_CONFIG = {
    "num_scenarios": 20,              # 기본값, stage에서 개별 덮어쓰기
    "traffic_mode": "trigger",
    "vehicle_config": {
        "lidar": {"num_lasers": 72, "distance": 50},
        "side_detector": {"num_lasers": 0},
        "lane_line_detector": {"num_lasers": 0},
    },
    "use_render": False,
    "decision_repeat": 1,             # 기본값, stage에서 덮어쓰기
    "horizon": 1000,                  # 기본값, stage에서 덮어쓰기

    # ===== Reward 설계 (안전 및 안정성 강화) =====
    # "driving_reward": 0.5,            # per-step 주행 보상 (너무 크지 않게)
    # "speed_reward": 0.1,
    # "out_of_road_penalty": 10.0,
    # "crash_vehicle_penalty": 10.0,
    # "crash_object_penalty": 10.0,
    # "crash_sidewalk_penalty": 10.0,
    # "success_reward": 120.0,          # 성공 시 확실히 튀도록 크게
    
    # ===== 초단순 보상 설계 (학습 가능성 최우선) =====
    "driving_reward": 3.0,              # 전진 보상 대폭 증가 (움직임 강력 장려)
    "speed_reward": 0.8,                # 속도 보상 증가 (빠르게 갈수록 이득)
    "out_of_road_penalty": 3.0,         # 페널티 최소화 (학습 초기 탐험 허용)
    "crash_vehicle_penalty": 5.0,       # 충돌 페널티 완화
    "crash_object_penalty": 3.0,        # 기물 페널티 완화
    "crash_sidewalk_penalty": 3.0,      # 보도 페널티 완화
    "use_lateral_reward": True,         # 차선 중앙 유지 보상 활성화
    "success_reward": 50.0,             # 성공 보상 (명확한 목표)
}

# PPO 알고리즘 설정 (MetaDrive Best Practice 기반 튜닝)
PPO_CONFIG = {
    "learning_rate": 5e-4,              # 3e-4 → 5e-4 (학습 속도 증가)
    "n_steps": 2048,                    # 4096 → 2048 (빠른 업데이트)
    "batch_size": 256,                  # 512 → 256 (빠른 학습)
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.02,                   # 0.01 → 0.02 (탐험 더 증가)
    "max_grad_norm": 0.5,               # 1.0 → 0.5 (안정성)
    "verbose": 1,
    "tensorboard_log": LOGS_DIR,
}

# ============================================================
# 8단계 커리큘럼 정의 (점진적 기술 습득)
# ============================================================

CURRICULUM_STAGES = {
    "stage_1": {
        "name": "Stage 1: Learn to Move Forward",
        "steps": 200_000,                    # 100k → 200k (충분한 학습 시간)
        "map": "S", "random_traffic": False, "traffic_density": 0.0,
        "num_scenarios": 5,                  # 10 → 5 (단순화)
        "decision_repeat": 1, "horizon": 1000,  # 600 → 1000 (더 긴 에피소드)
        "description": "직선 도로에서 전진 및 차선 유지 학습 (매우 단순)",
    },
    "stage_2": {
        "name": "Stage 2: Master Straight Driving",
        "steps": 300_000,                    # 150k → 300k (더 긴 학습)
        "map": "S", "random_traffic": False, "traffic_density": 0.0,
        "num_scenarios": 10,                 # 20 → 10
        "decision_repeat": 1, "horizon": 1200,  # 800 → 1200
        "description": "직선 도로 완벽 마스터, 안정적 차선 유지",
    },
    "stage_3": {
        "name": "Stage 3: Master a Single Gentle Curve",
        "steps": 200_000, "map": "C", "random_traffic": False, "traffic_density": 0.0,
        "num_scenarios": 30, "decision_repeat": 1, "horizon": 800,
        "description": "복잡한 전환 없이, '하나의 커브'를 안정적으로 도는 조향 제어 집중 학습.",
    },
    "stage_4": {
        "name": "Stage 4: Transition from Straight to Curve",
        "steps": 250_000, "map": "SC", "random_traffic": False, "traffic_density": 0.0,
        "num_scenarios": 50, "decision_repeat": 1, "horizon": 1000,
        "description": "직선 주행에서 커브 주행으로 부드럽게 '전환'하는 기술 학습.",
    },
    "stage_5": {
        "name": "Stage 5: Drive on Complex Roads (No Traffic)",
        "steps": 300_000, "map": "TSCO", "random_traffic": False, "traffic_density": 0.0,
        "num_scenarios": 80, "decision_repeat": 1, "horizon": 1200,
        "description": "다양한 교차로와 커브가 있는 복합 도로를 교통 없는 상태에서 완주하며 주행 기술 종합.",
    },
    "stage_6": {
        "name": "Stage 6: Avoid a Single Stationary Obstacle",
        "steps": 200_000, "map": "S", "random_traffic": False, "traffic_density": 0.05,
        "num_scenarios": 50, "decision_repeat": 1, "horizon": 1000,
        "description": "직선 도로 위 '정지된 차 한 대'를 인지하고 안전하게 회피하는 기본기 학습.",
    },
    "stage_7": {
        "name": "Stage 7: Handle Light Dynamic Traffic",
        "steps": 300_000, "map": "SC", "random_traffic": True, "traffic_density": 0.1,
        "num_scenarios": 100, "decision_repeat": 1, "horizon": 1500,
        "description": "움직이는 소수의 차량들 사이에서 안전하게 주행하는 능력 학습.",
    },
    "stage_8": {
        "name": "Stage 8: Final Challenge (Complex Map, Dense Traffic)",
        "steps": 400_000, "map": "TSCO", "random_traffic": True, "traffic_density": 0.15,
        "num_scenarios": 150, "decision_repeat": 2, "horizon": 2000,
        "description": "가장 복잡한 환경에서 모든 기술을 종합하여 최종 주행 능력 검증.",
    },
}

# ============================================================
# 전체 학습 설정
# ============================================================

TOTAL_STEPS = sum(stage["steps"] for stage in CURRICULUM_STAGES.values())

TRAINING_CONFIG = {
    "total_timesteps": TOTAL_STEPS,
    "model_name": "ppo_8_stage_curriculum_v1",  # 새 커리큘럼을 위한 모델 이름
    "save_freq": 50_000,
}

# ============================================================
# 정보 출력 함수
# ============================================================

def print_ppo_curriculum_info():
    """커리큘럼 설정 정보 출력"""
    print("\n" + "=" * 70)
    print(" PPO 5-Stage Curriculum Learning (v2)")
    print("=" * 70)
    print(f" Model Name       : {TRAINING_CONFIG['model_name']}")
    print(f" Total Timesteps  : {TOTAL_STEPS:,}")
    print(f" Models Directory : {MODELS_DIR}")
    print(f" Logs Directory   : {LOGS_DIR}")
    
    for stage_name, stage_config in CURRICULUM_STAGES.items():
        print("\n" + "-" * 30)
        print(f" {stage_config['name']} ({stage_name})")
        print(f"  ├─ Steps          : {stage_config['steps']:,}")
        print(f"  ├─ Map            : '{stage_config['map']}'")
        print(f"  ├─ Random Traffic : {stage_config['random_traffic']}")
        print(f"  ├─ Traffic Density: {stage_config['traffic_density']}")
        print(f"  ├─ Num Scenarios  : {stage_config['num_scenarios']}")
        print(f"  ├─ Decision Repeat: {stage_config['decision_repeat']}")
        print(f"  ├─ Horizon        : {stage_config['horizon']}")
        print(f"  └─ Goal           : {stage_config['description']}")
        
    print("\n" + "=" * 70 + "\n")

if __name__ == "__main__":
    print_ppo_curriculum_info()
