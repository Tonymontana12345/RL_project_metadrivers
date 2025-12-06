"""
PPO 커리큘럼 러닝 전용 설정 파일 (신규)

요청사항 기반 5단계 커리큘럼
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

# 모든 단계에 공통적으로 적용될 기본 환경 설정
BASE_ENV_CONFIG = {
    "num_scenarios": 1,
    "traffic_mode": "trigger",
    "vehicle_config": {
        "lidar": {"num_lasers": 72, "distance": 50},
        "side_detector": {"num_lasers": 0},
        "lane_line_detector": {"num_lasers": 0},
    },
    "use_render": False,
    "decision_repeat": 5, # 이 값은 train_ppo_curriculum.py에서 동적으로 덮어쓰게 됩니다.
    "horizon": 2000,
    "driving_reward": 1.0,
    "speed_reward": 0.1,
    "out_of_road_penalty": 5.0, # 이전 단계에서 수정한 값 유지
    "crash_vehicle_penalty": 10.0,
    "crash_object_penalty": 10.0,
    "crash_sidewalk_penalty": 10.0,
    "success_reward": 80.0,
}

# PPO 알고리즘 설정 (MetaDrive Best Practice)
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
# ============================================================ 

CURRICULUM_STAGES = {
    "stage_1": {
        "name": "Stage 1: Straight Road, No Traffic",
        "steps": 300000,
        "map": "S",
        "random_traffic": False,
        "traffic_density": 0.0,
        "description": "가장 기초적인 직진 주행 및 차선 유지 학습",
    },
    "stage_2": {
        "name": "Stage 2: Simple Curve, No Traffic",
        "steps": 300000,
        "map": "SC",
        "random_traffic": False,
        "traffic_density": 0.0,
        "description": "커브 길 주행 능력 학습",
    },
    "stage_3": {
        "name": "Stage 3: Complex Road, No Traffic",
        "steps": 300000,
        "map": "TSCO",
        "random_traffic": False,
        "traffic_density": 0.0,
        "description": "교차로, 원형 교차로 등 복잡한 맵 구조 주행 학습",
    },
    "stage_4": {
        "name": "Stage 4: Complex Road, Light Traffic",
        "steps": 300000,
        "map": "TSCO",
        "random_traffic": True,
        "traffic_density": 0.05,
        "description": "다른 차량이 있는 환경에서 회피 및 대응 능력 학습",
    },
    "stage_5": {
        "name": "Stage 5: Complex Road, Hard Traffic",
        "steps": 300000,
        "map": "TSCO",
        "random_traffic": True,
        "traffic_density": 0.10,
        "description": "복잡하고 밀도 높은 교통 상황에서의 최종 주행 능력 검증",
    },
}

# ============================================================ 
# 전체 학습 설정
# ============================================================ 

TOTAL_STEPS = sum(stage["steps"] for stage in CURRICULUM_STAGES.values())

TRAINING_CONFIG = {
    "total_timesteps": TOTAL_STEPS,
    "model_name": "ppo_5_stage_curriculum",
    "save_freq": 50000,
}

# ============================================================ 
# 정보 출력 함수
# ============================================================ 

def print_ppo_curriculum_info():
    """커리큘럼 설정 정보 출력"""
    print("\n" + "="*70)
    print(" PPO 5-Stage Curriculum Learning")
    print("="*70)
    print(f" Model Name: {TRAINING_CONFIG['model_name']}")
    print(f" Total Timesteps: {TOTAL_STEPS:,}")
    
    for stage_name, stage_config in CURRICULUM_STAGES.items():
        print("\n" + "-"*30)
        print(f" {stage_config['name']}")
        print(f"  ├─ Steps: {stage_config['steps']:,}")
        print(f"  ├─ Map: '{stage_config['map']}'")
        print(f"  ├─ Traffic Density: {stage_config['traffic_density']}")
        print(f"  └─ Goal: {stage_config['description']}")
        
    print("\n" + "="*70 + "\n")

if __name__ == "__main__":
    print_ppo_curriculum_info()
