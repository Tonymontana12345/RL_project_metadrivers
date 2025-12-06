"""
Path utility functions for the project.
"""

import os
from config import MODELS_DIR, LOGS_DIR, RESULTS_DIR

def get_model_path(model_name, timestep=None):
    """모델 파일 경로 생성"""
    if timestep:
        filename = f"{model_name}_{timestep}.zip"
    else:
        filename = f"{model_name}.zip"
    return os.path.join(MODELS_DIR, filename)

def get_log_path(experiment_name):
    """로그 디렉토리 경로 생성"""
    log_dir = os.path.join(LOGS_DIR, experiment_name)
    os.makedirs(log_dir, exist_ok=True)
    return log_dir

def get_result_path(filename):
    """결과 파일 경로 생성"""
    return os.path.join(RESULTS_DIR, filename)
