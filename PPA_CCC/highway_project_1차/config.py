"""
PGDrive 강화학습 프로젝트 설정 파일

모든 실험 설정을 여기서 관리합니다.
"""

import os

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

# 테스트용 시드들 (일반화 성능 평가)
TEST_SEEDS = [2000, 2001, 2002, 2003, 2004]

# 다중 시드 학습용 (비교 실험)
TRAIN_SEEDS = list(range(1000, 1010))  # 1000-1009 (10개)


# ============================================================ 
# 환경 설정
# ============================================================ 

# 고정 시드 환경 설정
FIXED_SEED_ENV_CONFIG = {
    # 시드 설정
    "start_seed": FIXED_SEED,
    "num_scenarios": 1,  # 1개 맵만 사용. 'environment_num'에서 변경
    
    # 맵 설정
    "map": 5,  # 5개 블록으로 구성된 맵
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
    
    # 보상 설정 (기본값 사용)
    "driving_reward": 1.0,
    "speed_reward": 0.1,
    "out_of_road_penalty": 5.0,
    "crash_vehicle_penalty": 5.0, # 'crash_penalty'에서 변경
    "crash_object_penalty": 5.0,  # 'crash_penalty'에서 변경
    "crash_sidewalk_penalty": 5.0,# 'crash_penalty'에서 변경
    "success_reward": 10.0,
}

# 다중 시드 환경 설정 (비교용)
MULTI_SEED_ENV_CONFIG = {
    **FIXED_SEED_ENV_CONFIG,
    "start_seed": TRAIN_SEEDS[0],
    "num_scenarios": len(TRAIN_SEEDS),  # 10개 맵. 'environment_num'에서 변경
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

# PPO 알고리즘 설정
PPO_CONFIG = {
    "learning_rate": 3e-4,
    "n_steps": 2048,
    "batch_size": 64,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.01,
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

# SAC 알고리즘 설정 (선택적)
SAC_CONFIG = {
    "learning_rate": 3e-4,
    "buffer_size": 100000,
    "learning_starts": 1000,
    "batch_size": 256,
    "tau": 0.005,
    "gamma": 0.99,
    "train_freq": 1,
    "gradient_steps": 1,
    "ent_coef": "auto",
    "verbose": 1,
    "tensorboard_log": LOGS_DIR,
}

# TD3 알고리즘 설정 (선택적)
TD3_CONFIG = {
    "learning_rate": 1e-3,
    "buffer_size": 100000,
    "learning_starts": 1000,
    "batch_size": 256,
    "tau": 0.005,
    "gamma": 0.99,
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

# 고정 시드 학습
FIXED_SEED_TRAINING = {
    "total_timesteps": 100000,  # 총 학습 스텝
    "save_freq": 10000,         # 모델 저장 주기
    "eval_freq": 5000,          # 평가 주기
    "n_eval_episodes": 10,      # 평가 에피소드 수
    "model_name": f"ppo_fixed_seed_{FIXED_SEED}",
}

# 다중 시드 학습 (비교용)
MULTI_SEED_TRAINING = {
    "total_timesteps": 200000,  # 더 많은 스텝 필요
    "save_freq": 20000,
    "eval_freq": 10000,
    "n_eval_episodes": 20,
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




