"""
커리큘럼 러닝 전용 설정 파일

목표: 충돌 없이 안전하게 완주
방법: 쉬운 환경 → 중간 환경 → 어려운 환경 (점진적 학습)
"""

import os
import random
import numpy as np
import torch


# ============================================================ 
# 랜덤 시드 고정 함수
# ============================================================ 

def set_global_seed(seed):
    """
    모든 랜덤 시드를 고정하여 재현성 보장
    
    Args:
        seed: 고정할 시드 값
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # 추가 재현성 보장
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ============================================================ 
# 프로젝트 경로
# ============================================================ 

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

# 디렉토리 생성
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================ 
# 랜덤 시드 설정
# ============================================================ 

# 고정 시드 (메인 학습용)
FIXED_SEED = 1000

# 학습용 시드 (균형잡힌 난이도) - 6개
# 난이도 분포: 쉬움 2개(6339:40%, 8576:100%), 중간 2개(1409:20%, 4806:20%), 어려움 2개(2824:0-10%, 5506:0-10%)
TRAIN_SEEDS = [1409, 2824, 5506, 6339, 8576, 4806]

# 평가용 시드 (학습 시드와 완전히 분리) - 5개
# 난이도 분포: 4657(35%), 9935(45%), 2679/3286/5012(0% → 10-20% 예상)
TEST_SEEDS = [2679, 3286, 4657, 5012, 9935]


# ============================================================ 
# 환경 설정 (안전 중심)
# ============================================================ 

# 기본 환경 설정 (커리큘럼의 기반)
BASE_ENV_CONFIG = {
    # 시드 설정
    "start_seed": FIXED_SEED,
    "num_scenarios": 1,
    
    # 맵 설정
    "map": 3,  # 3개 블록으로 구성된 맵
    
    # 트래픽 설정 (단계별로 변경됨)
    "random_traffic": True,
    "traffic_density": 0.05,  # 기본값 (단계별로 오버라이드)
    "traffic_mode": "trigger",
    
    # 차량 센서 설정 (관측 공간 최적화: 91차원)
    "vehicle_config": {
        "lidar": {
            "num_lasers": 72,
            "distance": 50,
        },
        "side_detector": {
            "num_lasers": 0
        },
        "lane_line_detector": {
            "num_lasers": 0
        },
    },
    
    # 렌더링 설정
    "use_render": False,
    
    # 액션 설정
    "decision_repeat": 5,
    
    # 에피소드 길이 제한
    "horizon": 2000,  # 기본값 (단계별로 오버라이드 가능)
    
    # ========================================
    # 보상 함수 (옵션 1: 균형 잡힌 안전 중심) ⭐⭐⭐
    # ========================================
    
    # 주행 보상 (복원 + 약간 조정)
    "driving_reward": 1.0,        # 0.5 → 1.0 (복원)
    "speed_reward": 0.1,          # 0.0 → 0.1 (복원! 중요!)
    
    # 실패 페널티 (적당히 강화)
    "out_of_road_penalty": 20.0,      # 40.0 → 20.0 (너무 크면 탐험 못함)
    "crash_vehicle_penalty": 20.0,    # 50.0 → 20.0 (너무 크면 탐험 못함)
    "crash_object_penalty": 20.0,     # 50.0 → 20.0 (너무 크면 탐험 못함)
    "crash_sidewalk_penalty": 20.0,   # 40.0 → 20.0 (너무 크면 탐험 못함)
    
    # 성공 보상 (적당히 증가)
    "success_reward": 50.0,       # 100.0 → 50.0 (적당히)
}


# ============================================================ 
# 커리큘럼 러닝 설정 ⭐⭐⭐⭐⭐
# ============================================================ 

CURRICULUM_CONFIG = {
    # 단계 1: 쉬운 환경 (기본 주행 학습)
    "stage_1": {
        "name": "Easy",
        "steps": 500000,              # 300k → 500k (더 충분히)
        "traffic_density": 0.05,      # 차량 거의 없음
        "horizon": 2500,              # 시간 여유
        "description": "기본 주행, 차선 유지, 목표 도달 학습",
        "expected_success_rate": "60-80% (쉬운 환경에서)",
    },
    
    # 단계 2: 중간 환경 (회피 기술 학습)
    "stage_2": {
        "name": "Medium",
        "steps": 500000,              # 300k → 500k (더 충분히)
        "traffic_density": 0.10,      # 적당한 교통량
        "horizon": 2000,              # 표준 시간
        "description": "차량 회피, 차선 변경, 복잡한 상황 대처",
        "expected_success_rate": "50-70% (중간 환경에서)",
    },
    
    # 단계 3: 어려운 환경 (마스터)
    "stage_3": {
        "name": "Hard",
        "steps": 500000,              # 300k → 500k (더 충분히)
        "traffic_density": 0.15,      # 많은 교통량
        "horizon": 2000,              # 표준 시간
        "description": "복잡한 교통, 돌발 상황, 일반화",
        "expected_success_rate": "60-80% (어려운 환경에서!) 🎯",
    },
    
    # 전체 설정
    "total_steps": 1500000,           # 900k → 1.5M steps (충분히!)
    "save_freq": 50000,               # 50k마다 저장
    "eval_freq": 50000,               # 50k마다 평가
}


# ============================================================ 
# PPO 설정 (안전 학습 최적화)
# ============================================================ 

PPO_CONFIG = {
    "learning_rate": 1e-4,        # 3e-4 → 1e-4 (더 안정적)
    "n_steps": 4096,              # 2048 → 4096 (긴 에피소드)
    "batch_size": 128,            # 64 → 128 (더 안정적)
    "n_epochs": 20,               # 10 → 20 (더 많이 학습)
    "gamma": 0.998,               # 0.995 → 0.998 (장기 보상 중시!)
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.02,             # 0.01 → 0.02 (탐험 증가)
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "verbose": 1,
    "tensorboard_log": LOGS_DIR,
}


# ============================================================ 
# 커리큘럼 학습 설정
# ============================================================ 

CURRICULUM_TRAINING = {
    "total_timesteps": CURRICULUM_CONFIG["total_steps"],
    "save_freq": CURRICULUM_CONFIG["save_freq"],
    "eval_freq": CURRICULUM_CONFIG["eval_freq"],
    "n_eval_episodes": 20,
    "model_name": "ppo_curriculum_v6_balanced",  # v5 → v6 (보상 함수 수정)
    "use_curriculum": True,
}


# ============================================================ 
# 평가 설정
# ============================================================ 

EVALUATION_CONFIG = {
    "n_eval_episodes": 10,
    "deterministic": True,
    "render": False,
    "save_video": False,
}


# ============================================================ 
# 시각화 설정
# ============================================================ 

PLOT_CONFIG = {
    "figsize": (12, 6),
    "dpi": 100,
    "style": "seaborn-v0_8",
    "save_format": "png",
}


# ============================================================ 
# 출력용 정보
# ============================================================ 

def print_curriculum_info():
    """커리큘럼 설정 정보 출력"""
    print("\n" + "="*70)
    print("📚 커리큘럼 러닝 설정 정보")
    print("="*70)
    
    print("\n🎯 목표: 충돌 없이 안전하게 완주!")
    print("   - Safety Score: 95%+ (충돌/도로이탈 최소)")
    print("   - Success Rate: 70%+ (목표 도달)")
    print("   - Overall Score: 85%+ (종합 점수)")
    
    print("\n📊 보상 함수 (균형 잡힌 안전 중심):")
    print(f"   - Driving Reward: {BASE_ENV_CONFIG['driving_reward']}")
    print(f"   - Speed Reward: {BASE_ENV_CONFIG['speed_reward']} (복원!)")
    print(f"   - Crash Penalty: {BASE_ENV_CONFIG['crash_vehicle_penalty']} (2배 증가)")
    print(f"   - Out of Road Penalty: {BASE_ENV_CONFIG['out_of_road_penalty']} (2배 증가)")
    print(f"   - Success Reward: {BASE_ENV_CONFIG['success_reward']} (2.5배 증가)")
    
    print("\n📚 커리큘럼 단계:")
    for stage_name, stage_config in CURRICULUM_CONFIG.items():
        if stage_name in ["stage_1", "stage_2", "stage_3"]:
            print(f"\n   {stage_name.upper()}: {stage_config['name']}")
            print(f"   ├─ 학습 스텝: {stage_config['steps']:,}")
            print(f"   ├─ Traffic Density: {stage_config['traffic_density']}")
            print(f"   ├─ Horizon: {stage_config['horizon']}")
            print(f"   ├─ 목표: {stage_config['description']}")
            print(f"   └─ 예상 성능: {stage_config['expected_success_rate']}")
    
    print(f"\n⏱️  총 학습 시간: 약 5시간 (1.5M steps)")
    print(f"💾 체크포인트: 50k마다 저장")
    print(f"📈 평가: 50k마다 평가")
    
    print("\n🎓 학습 시드: " + str(TRAIN_SEEDS))
    print("🧪 테스트 시드: " + str(TEST_SEEDS))
    
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    # 설정 정보 출력
    print_curriculum_info()

