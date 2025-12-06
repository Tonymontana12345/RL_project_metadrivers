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

    # ===== Reward 설계 (PPO가 '성공'을 확실히 구분하도록 스케일 조정) =====
    "driving_reward": 0.5,            # per-step 주행 보상 (너무 크지 않게)
    "speed_reward": 0.1,
    "out_of_road_penalty": 10.0,
    "crash_vehicle_penalty": 10.0,
    "crash_object_penalty": 10.0,
    "crash_sidewalk_penalty": 10.0,
    "success_reward": 120.0,          # 성공 시 확실히 튀도록 크게
}

# PPO 알고리즘 설정 (MetaDrive Best Practice 기반 튜닝)
PPO_CONFIG = {
    "learning_rate": 3e-4,
    "n_steps": 2048,
    "batch_size": 256,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.005,
    "max_grad_norm": 1.0,
    "verbose": 1,
    "tensorboard_log": LOGS_DIR,
}

# ============================================================
# 5단계 커리큘럼 정의
#   - 난이도 점진적 증가
#   - Stage별 horizon / decision_repeat / num_scenarios 세밀 튜닝
# ============================================================

CURRICULUM_STAGES = {
    "stage_1": {
        "name": "Stage 1: Straight Road, No Traffic",
        "steps": 200_000,
        "map": "S",
        "random_traffic": False,
        "traffic_density": 0.0,
        "num_scenarios": 10,
        "decision_repeat": 1,
        "horizon": 600,
        "description": "직선 도로 + 무트래픽. 기본 주행 및 차선 유지 학습 단계.",
    },
    "stage_2": {
        "name": "Stage 2: Simple Curve, No Traffic",
        "steps": 250_000,
        "map": "SC",
        "random_traffic": False,
        "traffic_density": 0.0,
        "num_scenarios": 20,
        "decision_repeat": 1,
        "horizon": 800,
        "description": "완만한 커브 구간에서 조향 제어 학습.",
    },
    "stage_3": {
        "name": "Stage 3: Complex Road (TSC), No Traffic",
        "steps": 300_000,
        "map": "TSC",               # ⚠️ TSCO 대신 TSC로 난이도 one-step down
        "random_traffic": False,
        "traffic_density": 0.0,
        "num_scenarios": 40,
        "decision_repeat": 1,
        "horizon": 1000,
        "description": "복잡하지만 라운드어바웃은 없는 TSC 맵에서 복잡 경로 학습.",
    },
    "stage_4": {
        "name": "Stage 4: TSCO, Light Traffic",
        "steps": 400_000,
        "map": "TSCO",
        "random_traffic": True,
        "traffic_density": 0.05,
        "num_scenarios": 80,
        "decision_repeat": 1,       # TSCO + 약한 트래픽, 여전히 프레임당 제어 강조
        "horizon": 1200,
        "description": "라운드어바웃 포함 TSCO 맵 + 약한 트래픽 환경.",
    },
    "stage_5": {
        "name": "Stage 5: TSCO, Hard Traffic",
        "steps": 500_000,
        "map": "TSCO",
        "random_traffic": True,
        "traffic_density": 0.10,
        "num_scenarios": 150,
        "decision_repeat": 2,       # 최종 고난도 환경에서 약간의 action repeat 허용
        "horizon": 1500,
        "description": "TSCO + 고밀도 트래픽. 최종 정책 성능 검증 단계.",
    },
}

# ============================================================
# 전체 학습 설정
# ============================================================

TOTAL_STEPS = sum(stage["steps"] for stage in CURRICULUM_STAGES.values())

TRAINING_CONFIG = {
    "total_timesteps": TOTAL_STEPS,
    "model_name": "ppo_5_stage_curriculum_v2",
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
