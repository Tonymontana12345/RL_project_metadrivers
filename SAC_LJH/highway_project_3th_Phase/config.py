"""
PGDrive 강화학습 프로젝트 설정 파일

모든 실험 설정을 여기서 관리합니다.
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
# 환경 설정
# ============================================================ 

# 고정 시드 환경 설정
FIXED_SEED_ENV_CONFIG = {
    # 시드 설정
    "start_seed": FIXED_SEED,
    "num_scenarios": 1,  # 1개 맵만 사용
    
    # 맵 설정
    "map": "TSCO",  # TSCO 블록 타입: T-intersection → Straight → Circular → Roundabout
    
    # 트래픽 설정
    "random_traffic": True,   # 다양한 트래픽 패턴 (일반화 향상)
    "traffic_density": 0.1,   # 기본값으로 복귀 (난이도 낮춤)
    "traffic_mode": "trigger",  # trigger 또는 reborn

    # 차량 센서 설정 (관측 공간 최적화)
    "vehicle_config": {
        "lidar": {
            "num_lasers": 72,    # 72개 (5° 간격) - 학습 속도/성능 균형 (실험 결과 120보다 우수)
            "distance":70,      # 70m 감지 거리 (마주오는 차량 조기 감지 - 좌회전 충돌 방지) 
        },
        "side_detector": {
            "num_lasers": 2      # 측면 감지기 활성화 (좌/우 차량 감지). -->2로 변경시 제일 좋음. 차원 문제 때문에 앙상블시 축소 
            
        },
        "lane_line_detector": {
            "num_lasers": 2      # 차선 감지기 활성화 (좌/우 차선 거리) -->2로 변경시 제일 좋음.차원 문제 때문에 앙상블시 축소 
        },
    },
    
    # 렌더링 설정
    "use_render": False,  # 학습 시 False (빠름)
    
    # 액션 설정
    "decision_repeat":5,  # 1,2차 5 -->2 액션 반복 (5 * 0.02s = 0.1s)
                          # 3차부터 5 
    
    # 에피소드 길이 제한 (중요!)
    "horizon": 1000,  # 최대 1000 스텝 (너무 길면 길 잃고 이탈, TSCO 맵은 짧게)
    
    # 보상 설정 (안전 강조) 1,2차
    # "driving_reward": 1.0,
    # "speed_reward": 0.1,
    # "out_of_road_penalty": 10.0,      # 5.0 → 10.0 (안전 강조)
    # "crash_vehicle_penalty": 10.0,    # 5.0 → 10.0 (안전 강조)
    # "crash_object_penalty": 10.0,     # 5.0 → 10.0 (안전 강조)
    # "crash_sidewalk_penalty": 10.0,   # 5.0 → 10.0 (안전 강조)
    # "success_reward": 20.0,           # 10.0 → 20.0 (목표 강조)
   #3차
    # "use_lateral_reward": True,
    # "driving_reward": 1.5,
    # "speed_reward": 0.1,
    # "out_of_road_penalty": 7.0,      # 5.0 → 10.0 (안전 강조)
    # "crash_vehicle_penalty": 7.0,    # 5.0 → 10.0 (안전 강조)
    # "crash_object_penalty": 10.0,     # 5.0 → 10.0 (안전 강조)
    # "crash_sidewalk_penalty": 10.0,   # 5.0 → 10.0 (안전 강조)
    # "success_reward": 100.0,           # 10.0 → 20.0 (목표 강조)
     #4차
    # "use_lateral_reward": false,
    # "driving_reward": 1.5,
    # "speed_reward": 0.1,
    # "out_of_road_penalty": 7.0,      # 5.0 → 10.0 (안전 강조)
    # "crash_vehicle_penalty": 7.0,    # 5.0 → 10.0 (안전 강조)
    # "crash_object_penalty": 10.0,     # 5.0 → 10.0 (안전 강조)
    # "crash_sidewalk_penalty": 10.0,   # 5.0 → 10.0 (안전 강조)
    # "success_reward": 200.0,           # 10.0 → 20.0 (목표 강조) 

    # 보상 함수 - 기본 설정 (MetaDrive 표준)
    "driving_reward": 1.0,            # 목표 방향 주행
    "speed_reward": 0.1,              # 속도 유지
    "out_of_road_penalty": 5.0,       # 도로 이탈
    "crash_vehicle_penalty": 5.0,     # 차량 충돌
    "crash_object_penalty": 5.0,      # 물체 충돌
    "crash_sidewalk_penalty": 5.0,    # 인도 충돌
    "success_reward": 10.0,           # 목표 도달

}

# 다중 시드 환경 설정 (3개 랜덤 시드)
MULTI_SEED_ENV_CONFIG = {
    **FIXED_SEED_ENV_CONFIG,
    "start_seed": TRAIN_SEEDS[0],
    "num_scenarios": len(TRAIN_SEEDS),  # 3개 맵 (1409, 2824, 5506)
}

# 평가용 환경 설정 (렌더링 켜기)
EVAL_ENV_CONFIG = {
    **FIXED_SEED_ENV_CONFIG,
    "use_render": True,  # 시각화
    "force_fps": 30,     # FPS 제한
}

# ============================================================
# 커리큘럼 러닝 환경 설정 (단계별 난이도 증가)
# ============================================================

# Stage 1: 로터리만 (로터리 집중 학습!)
STAGE1_ENV_CONFIG = {
    **FIXED_SEED_ENV_CONFIG,
    "map": "O",  # rOundabout (로터리만)    
    "traffic_density": 0.05,  # 매우 낮은 트래픽
    "horizon": 1500,  # 로터리는 긴 시간 필요
}

# Stage 2: 로터리 + 곡선
STAGE2_ENV_CONFIG = {
    **FIXED_SEED_ENV_CONFIG,
    "map": "OC",  # rOundabout + Curve
    "num_scenarios":5,
    "traffic_density": 0.08,  # 낮은 트래픽
    "horizon": 1500,
}

# Stage 3: T교차로 + 로터리 + 곡선
STAGE3_ENV_CONFIG = {
    **FIXED_SEED_ENV_CONFIG,
    "map": "TOC",  # T-intersection + rOundabout + Curve
    "traffic_density": 0.08,  # 낮은 트래픽
    "num_scenarios":10,
    "horizon": 1500,
}

# Stage 4: 최종 맵 (TOCS - 전체 요소)
STAGE4_ENV_CONFIG = {
    **FIXED_SEED_ENV_CONFIG,
    "map": "TOCS",  # T + rOundabout + Curve + Straight
    "traffic_density": 0.1,  # 중간 트래픽
    "num_scenarios":20,
    "horizon": 1500,
}

# Stage 5: 십자교차로 추가 (TSCOX)
STAGE5_ENV_CONFIG = {
    **FIXED_SEED_ENV_CONFIG,
    "map": "TSCOX",  # + X-intersection (십자교차로)
    "traffic_density": 0.1,
    "num_scenarios": 30,    # 복잡도 증가로 시나리오 증가
    "horizon": 1500,
}

# Stage 6: 램프 추가 (TSCOXrR)
STAGE6_ENV_CONFIG = {
    **FIXED_SEED_ENV_CONFIG,
    "map": "TSCOXrR",  # + InRamp + OutRamp
    "traffic_density": 0.12,  # 트래픽 약간 증가
    "num_scenarios": 50,      # 더 많은 변형
    "horizon": 1500,
}


# ============================================================
# 학습 설정
# ============================================================ 

# PPO 알고리즘 설정 (자율주행 최적화)
PPO_CONFIG = {
    "learning_rate": 1e-4,      # 3e-4 → 1e-4 (더 안정적)
    "n_steps": 4096,            # 2048 → 4096 (긴 에피소드)
    "batch_size": 128,          # 64 → 128 (더 안정적)
    "n_epochs": 20,             # 10 → 20 (더 많이 학습)
    "gamma": 0.995,             # 0.99 → 0.995 (장기 보상)
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.001,          # 0.01 → 0.001 (탐험 줄임)
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "verbose": 1,
    "tensorboard_log": LOGS_DIR,
}

# DQN 알고리즘 설정 (선택적)
DQN_CONFIG = {
    "learning_rate": 1e-4,
    "buffer_size": 50000,
    "learning_starts": 1000,
    "batch_size": 32,
    "tau": 1.0,
    "gamma": 0.99,
    "train_freq": 4,
    "target_update_interval": 1000,
    "exploration_fraction": 0.1,
    "exploration_initial_eps": 1.0,
    "exploration_final_eps": 0.05,
    "verbose": 1,
    "tensorboard_log": LOGS_DIR,
}

# SAC 알고리즘 설정 (자율주행 최적화)
# SAC_CONFIG = {
#     "learning_rate": 3e-4,
#     "buffer_size": 500000,      # 100k → 500k (1M 스텝 대응)
#     "learning_starts": 5000,    # 1k → 5k (더 많은 초기 데이터)
#     "batch_size": 256,
#     "tau": 0.005,
#     "gamma": 0.995,             # 0.99 → 0.995 (장기 보상)
#     "train_freq": 1,
#     "gradient_steps": 1,
#     "ent_coef": "auto",
#     "verbose": 1,
#     "tensorboard_log": LOGS_DIR,
# }
#2차 
# SAC_CONFIG = {
#     "learning_rate": 3e-4,
#     "buffer_size": 100000,      # 100k → 500k (1M 스텝 대응)
#     "learning_starts": 5000,    # 1k → 5k (더 많은 초기 데이터)
#     "batch_size": 128,
#     "tau": 0.005,
#     "gamma": 0.97,             # 0.99 → 0.995 (장기 보상)
#     "train_freq": 1,
#     "gradient_steps": 1,
#     "ent_coef": 0.05,
#     "verbose": 1,
#     "tensorboard_log": LOGS_DIR,
# }
#3차 
# SAC_CONFIG = {
#     "learning_rate": 3e-4,
#     "buffer_size": 500000,      # 700k → 500k (충분함)
#     "learning_starts": 5000,    # 초기 탐험
#     "batch_size": 256,
#     "tau": 0.005,
#     "gamma": 0.98,              # 0.99 → 0.98 (커브에서 즉각 보상 중시)
#     "train_freq": 1,
#     "gradient_steps": 1,        # 2 → 1 (더 안정적)
#     "ent_coef": 0.1,            # "auto" → 0.1 (더 많은 탐험)
#     "verbose": 1,
#     "tensorboard_log": LOGS_DIR,
# }
# #4차
# SAC_CONFIG = {
#     "learning_rate": 3e-4,
#     "buffer_size": 500000,      # 700k → 500k (충분함)
#     "learning_starts": 5000,    # 초기 탐험
#     "batch_size": 128,
#     "tau": 0.005,
#     "gamma": 0.95,              # 0.99 → 0.98 (커브에서 즉각 보상 중시)
#     "train_freq": 1,
#     "gradient_steps": 1,        # 2 → 1 (더 안정적)
#     "ent_coef": 0.2,          # "auto" → 0.2 (더 많은 탐험)
#     "verbose": 1,
#     "tensorboard_log": LOGS_DIR,
# }
#5차
# SAC_CONFIG = {
#     "learning_rate": 3e-4,
#     "buffer_size": 200000,      # 700k → 500k (충분함)
#     "learning_starts": 5000,    # 초기 탐험
#     "batch_size": 128,
#     "tau": 0.005,
#     "gamma": 0.95,              # 0.99 → 0.98 (커브에서 즉각 보상 중시)
#     "train_freq": 1,
#     "gradient_steps": 1,        # 2 → 1 (더 안정적)
#     "ent_coef": 0.2,          # "auto" → 0.2 (더 많은 탐험)
#     "verbose": 1,
#     "tensorboard_log": LOGS_DIR,
# }
#6차
# SAC_CONFIG = {
#     "learning_rate": 3e-4,
#     "buffer_size": 300000,      # 700k → 500k (충분함)
#     "learning_starts": 10000,    # 초기 탐험
#     "batch_size": 128,
#     "tau": 0.005,
#     "gamma": 0.95,              # 0.99 → 0.98 (커브에서 즉각 보상 중시)
#     "train_freq": 1,
#     "gradient_steps": 1,        # 2 → 1 (더 안정적)
#     "ent_coef": "auto",          # "auto" → 0.2 (더 많은 탐험)
#     "verbose": 1,
#     "tensorboard_log": LOGS_DIR,
# }
#7차 - 로터리 멈춤 해결
# SAC_CONFIG = {
#     "learning_rate": 3e-4,
#     "buffer_size": 300000,
#     "learning_starts": 10000,
#     "batch_size": 128,
#     "tau": 0.005,
#     "gamma": 0.95,              # 즉각 보상 중시 (멈추면 바로 페널티)
#     "train_freq": 1,
#     "gradient_steps": 1,
#     "ent_coef": 0.15,           # "auto" → 0.15 (고정, 로터리에서 탐험 유지)
#     "verbose": 1,
#     "tensorboard_log": LOGS_DIR,
# }
#8차 - 옆차 밀기 방지 (보상 밸런스 조정)
# SAC_CONFIG = {
#     "learning_rate": 3e-4,
#     "buffer_size": 300000,
#     "learning_starts": 10000,
#     "batch_size": 128,
#     "tau": 0.005,
#     "gamma": 0.95,              # 즉각 보상 중시
#     "train_freq": 1,
#     "gradient_steps": 1,
#     "ent_coef": 0.1,            # 0.15 → 0.1 (탐험 감소, 안전 주행 학습)
#     "verbose": 1,
#     "tensorboard_log": LOGS_DIR,
# }
#9차 - 옆차 밀기 + 로터리 멈춤 동시 해결
# SAC_CONFIG = {
#     "learning_rate": 3e-4,
#     "buffer_size": 300000,
#     "learning_starts": 10000,
#     "batch_size": 128,
#     "tau": 0.005,
#     "gamma": 0.95,              # 즉각 보상 중시
#     "train_freq": 1,
#     "gradient_steps": 1,
#     "ent_coef": 0.12,           # 0.1 → 0.12 (약간 탐험 증가, 로터리 통과 유도)
#     "verbose": 1,
#     "tensorboard_log": LOGS_DIR,
# }
#10차 - 기본 설정으로 리셋 (TSCO 맵 최적화)
# Off-Policy 최적화: buffer_size 증가 (200k → 500k)  
#######점진적 알고리즘 전 단일 Test시 2차 값이 제일 좋음. 
SAC_CONFIG = {
    "learning_rate": 3e-4,
    "buffer_size": 500000,      # 200k → 500k (커리큘럼 학습 대응, Off-Policy 활용)
    "learning_starts": 5000,    # 10k → 5k (빠른 학습 시작)
    "batch_size": 256,          # 128 → 256 (더 안정적 학습)
    "tau": 0.005,
    "gamma": 0.95,              # 0.95 → 0.99 (장기 보상 중시)
    "train_freq": 1,
    "gradient_steps": 1,
    "ent_coef": "auto",         # 자동 조절 (초기 탐험 후 점진적 감소)
    "verbose": 1,
    "tensorboard_log": LOGS_DIR,
}
# TD3 알고리즘 설정 (자율주행 최적화)
TD3_CONFIG = {
    "learning_rate": 1e-3,
    "buffer_size": 500000,      # 100k → 500k (1M 스텝 대응)
    "learning_starts": 5000,    # 1k → 5k (더 많은 초기 데이터)
    "batch_size": 256,
    "tau": 0.005,
    "gamma": 0.995,             # 0.99 → 0.995 (장기 보상)
    "train_freq": 1,
    "gradient_steps": 1,
    "policy_delay": 2,
    "target_policy_noise": 0.2,
    "target_noise_clip": 0.5,
    "verbose": 1,
    "tensorboard_log": LOGS_DIR,
}


# ============================================================ 
# 학습 파라미터
# ============================================================ 

# 고정 시드 학습 (권장 설정)
FIXED_SEED_TRAINING = {
    "total_timesteps": 1000000,  # 100k → 1M (자율주행 권장)
    "save_freq": 50000,          # 10k → 50k (20회 저장)
    "eval_freq": 25000,          # 5k → 25k (40회 평가)
    "n_eval_episodes": 10,       # 평가 에피소드 수
    "model_name": f"ppo_fixed_seed_{FIXED_SEED}",
}

# 다중 시드 학습 (3개 랜덤 시드)
MULTI_SEED_TRAINING = {
    #"total_timesteps": 1000000,  # 200k → 1M (권장)
    "total_timesteps": 300000,  # 200k → 1M (권장)
    "save_freq": 50000,          # 20k → 50k
    "eval_freq": 25000,          # 10k → 25k
    "n_eval_episodes": 20,       # 평가 에피소드 수
    "model_name": "sac_multi_seed",
}

# 빠른 테스트용 (디버깅)
QUICK_TEST_TRAINING = {
    "total_timesteps": 10000,
    "save_freq": 5000,
    "eval_freq": 2500,
    "n_eval_episodes": 3,
    "model_name": "ppo_quick_test",
}

# ============================================================
# 커리큘럼 러닝 학습 설정 (단계별 학습)
# ============================================================

# Stage 1: 로터리만 (150k 스텝, ~30분)
STAGE1_TRAINING = {
    "total_timesteps": 150000,
    "save_freq": 75000,
    "model_name": "sac_stage1_roundabout",
}

# Stage 2: 로터리 + 곡선 (150k 스텝, ~30분)
STAGE2_TRAINING = {
    "total_timesteps": 250000,
    "save_freq": 75000,
    "model_name": "sac_stage2_roundabout_curve",
}

# Stage 3: T교차로 + 로터리 + 곡선 (200k 스텝, ~40분)
STAGE3_TRAINING = {
    "total_timesteps": 250000,
    "save_freq": 100000,
    "model_name": "sac_stage3_toc",
}

# Stage 4: 최종 맵 (300k 스텝, ~60분)
STAGE4_TRAINING = {
    "total_timesteps": 300000,
    "save_freq": 150000,
    "model_name": "sac_stage4_final",
}

# Stage 5: 십자교차로 추가 (300k 스텝)
STAGE5_TRAINING = {
    "total_timesteps": 300000,
    "save_freq": 150000,
    "model_name": "sac_stage5_intersection",
}

# Stage 6: 램프 추가 (300k 스텝)
STAGE6_TRAINING = {
    "total_timesteps": 300000,
    "save_freq": 150000,
    "model_name": "sac_stage6_ramps",
}


# ============================================================ 
# 평가 설정
# ============================================================ 

EVALUATION_CONFIG = {
    "n_eval_episodes": 10,
    "deterministic": True,       # 결정적 정책 사용
    "render": False,             # 렌더링 여부
    "save_video": True,         # 비디오 저장 여부
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
# 실험 시나리오
# ============================================================ 

EXPERIMENTS = {
    # 실험 1: 고정 시드 학습
    "exp1_fixed_seed": {
        "description": "시드 1000에서만 학습",
        "train_config": FIXED_SEED_TRAINING,
        "env_config": FIXED_SEED_ENV_CONFIG,
        "test_seeds": [FIXED_SEED] + TEST_SEEDS,
    },
    
    # 실험 2: 다중 시드 학습
    "exp2_multi_seed": {
        "description": "시드 1000-1009에서 학습",
        "train_config": MULTI_SEED_TRAINING,
        "env_config": MULTI_SEED_ENV_CONFIG,
        "test_seeds": TEST_SEEDS,
    },
    
    # 실험 3: 빠른 테스트
    "exp3_quick_test": {
        "description": "빠른 테스트 (디버깅용)",
        "train_config": QUICK_TEST_TRAINING,
        "env_config": FIXED_SEED_ENV_CONFIG,
        "test_seeds": [FIXED_SEED, TEST_SEEDS[0]],
    },
}
