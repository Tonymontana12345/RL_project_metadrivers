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
    "traffic_density": 0.15,  # 0.1 → 0.15 (더 어려운 환경)
    "traffic_mode": "trigger",  # trigger 또는 reborn
    
    # 차량 센서 설정 (관측 공간 최적화: 91차원)
    "vehicle_config": {
        "lidar": {
            "num_lasers": 72,    # 240 → 72 (차원 축소)
            "distance": 50,      # 감지 거리 50m
        },
        "side_detector": {
            "num_lasers": 0      # 비활성화 (불필요)
        },
        "lane_line_detector": {
            "num_lasers": 0      # 비활성화 (불필요)
        },
    },
    
    # 렌더링 설정
    "use_render": False,  # 학습 시 False (빠름)
    
    # 액션 설정
    "decision_repeat": 5,  # 액션 반복 (5 * 0.02s = 0.1s)
    
    # 에피소드 길이 제한 (중요!)
    "horizon": 2000,  # 최대 2000 스텝 (목표 도달에 충분한 시간)
    
    # 보상 설정 (안전 강조)
    "driving_reward": 1.0,
    "speed_reward": 0.1,
    "out_of_road_penalty": 10.0,      # 5.0 → 10.0 (안전 강조)
    "crash_vehicle_penalty": 10.0,    # 5.0 → 10.0 (안전 강조)
    "crash_object_penalty": 10.0,     # 5.0 → 10.0 (안전 강조)
    "crash_sidewalk_penalty": 10.0,   # 5.0 → 10.0 (안전 강조)
    "success_reward": 20.0,           # 10.0 → 20.0 (목표 강조)
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
SAC_CONFIG = {
    "learning_rate": 3e-4,
    "buffer_size": 500000,      # 100k → 500k (1M 스텝 대응)
    "learning_starts": 5000,    # 1k → 5k (더 많은 초기 데이터)
    "batch_size": 256,
    "tau": 0.005,
    "gamma": 0.995,             # 0.99 → 0.995 (장기 보상)
    "train_freq": 1,
    "gradient_steps": 1,
    "ent_coef": "auto",
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
    "total_timesteps": 500000,   # TSCO 맵 학습용 (500k)
    "save_freq": 50000,          # 10k → 50k (20회 저장)
    "eval_freq": 25000,          # 5k → 25k (40회 평가)
    "n_eval_episodes": 10,       # 평가 에피소드 수
    "model_name": "ppo_tsco_map_500k",  # TSCO 맵 학습 모델
}

# 다중 시드 학습 (6개 시드, 균형잡힌 난이도)
MULTI_SEED_TRAINING = {
    "total_timesteps": 800000,   # 800k (과적합 방지, 750k-800k 최적 구간)
    "save_freq": 50000,          # 50k마다 저장
    "eval_freq": 25000,          # 25k마다 평가
    "n_eval_episodes": 20,       # 평가 에피소드 수
    "model_name": "ppo_balanced_6seeds_v4_800k",  # v4: 균형잡힌 시드 구성
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
# 평가 설정
# ============================================================ 

EVALUATION_CONFIG = {
    "n_eval_episodes": 10,
    "deterministic": True,       # 결정적 정책 사용
    "render": False,             # 렌더링 여부
    "save_video": False,         # 비디오 저장 여부
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




