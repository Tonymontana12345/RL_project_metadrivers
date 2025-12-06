"""
Metadrive environment creation and wrapping.
"""

import gymnasium as gym
from metadrive.envs import MetaDriveEnv
from config import FIXED_SEED_ENV_CONFIG

def make_env(seed=None, render=False, config=None):
    """
    MetaDrive 환경 생성
    
    Args:
        seed: 랜덤 시드
        render: 렌더링 여부
        config: 환경 설정 dict (None이면 FIXED_SEED_ENV_CONFIG 사용)
    
    Returns:
        callable: 환경을 생성하는 함수
    
    Examples:
        # 기본 사용 (고정 시드 환경)
        env = make_env(seed=1000)()
        
        # 다중 시드 환경
        from config import MULTI_SEED_ENV_CONFIG
        env = make_env(seed=1000, config=MULTI_SEED_ENV_CONFIG)()
        
        # 커스텀 환경
        custom_config = {"map": 7, "traffic_density": 0.2}
        env = make_env(seed=1000, config=custom_config)()
    """
    # 설정 선택 (기본값: FIXED_SEED_ENV_CONFIG)
    if config is None:
        env_config = FIXED_SEED_ENV_CONFIG.copy()
    else:
        env_config = config.copy()
    
    # 시드 설정
    if seed is not None:
        env_config["start_seed"] = seed
    
    # 렌더링 설정
    if render:
        env_config["use_render"] = True
    
    def _init():
        env = MetaDriveEnv(config=env_config)
        return env
    
    return _init
