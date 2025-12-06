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

# 학습용 시드 (랜덤 선정, 고정) - 3개
TRAIN_SEEDS = [1409, 2824, 5506]

# 평가용 시드 (랜덤 선정, 고정) - 5개
TEST_SEEDS = [2679, 3286, 4657, 5012, 9935]


# ============================================================ 
# 환경 설정
# ============================================================ 

# 고정 시드 환경 설정
FIXED_SEED_ENV_CONFIG = {
    # 시드 설정
    "start_seed": FIXED_SEED,
    "num_scenarios": 1,  # 1개 맵만 사용. 'environment_num'에서 변경
    
    # 맵 설정
    "map": 3,  # 5개 블록으로 구성된 맵
    # "map": "SCSCS",  # 또는 특정 블록 시퀀스 지정
    
    # 트래픽 설정
    "random_traffic": False,  # 트래픽도 고정 (재현성)
    "traffic_density": 0.1,   # 차량 밀도 (10m당 차량 수)
    "traffic_mode": "trigger",  # trigger 또는 reborn
    
    # 렌더링 설정
    "use_render": False,  # 학습 시 False (빠름)
    # "offscreen_render": False, # 더 이상 사용되지 않음
    
    # 액션 설정
    "decision_repeat": 5,  # 액션 반복 (5 * 0.02s = 0.1s)
    
    # 에피소드 길이 제한 (중요!)
    "horizon": 2000,  # 최대 2000 스텝 (약 200초, 충분히 긴 시간)
    
    # 보상 설정 (기본값 사용)
    "driving_reward": 1.0,
    "speed_reward": 0.1,
    "out_of_road_penalty": 5.0,
    "crash_vehicle_penalty": 5.0, # 'crash_penalty'에서 변경
    "crash_object_penalty": 5.0,  # 'crash_penalty'에서 변경
    "crash_sidewalk_penalty": 5.0,# 'crash_penalty'에서 변경
    "success_reward": 10.0,
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
    "total_timesteps": 1000000,  # 100k → 1M (자율주행 권장)
    "save_freq": 50000,          # 10k → 50k (20회 저장)
    "eval_freq": 25000,          # 5k → 25k (40회 평가)
    "n_eval_episodes": 10,       # 평가 에피소드 수
    "model_name": f"ppo_fixed_seed_{FIXED_SEED}",
}

# 다중 시드 학습 (3개 랜덤 시드)
MULTI_SEED_TRAINING = {
    "total_timesteps": 1000000,  # 200k → 1M (권장)
    "save_freq": 50000,          # 20k → 50k
    "eval_freq": 25000,          # 10k → 25k
    "n_eval_episodes": 20,       # 평가 에피소드 수
    "model_name": "ppo_multi_seed",
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




