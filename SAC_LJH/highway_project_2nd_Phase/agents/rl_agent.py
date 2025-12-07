"""
RL Agent Factory

다양한 강화학습 알고리즘(PPO, SAC, TD3)의 모델을 생성하는 팩토리 모듈입니다.
"""

from stable_baselines3 import PPO, SAC, TD3
from config import PPO_CONFIG, SAC_CONFIG, TD3_CONFIG


def create_model(env, algorithm="ppo"):
    """
    Create an RL model with the specified algorithm.
    
    Args:
        env: The environment to train on.
        algorithm: Algorithm to use ("ppo", "sac", "td3")
    
    Returns:
        Model instance (PPO, SAC, or TD3)
    
    Raises:
        ValueError: If algorithm is not supported
    """
    algorithm = algorithm.lower()
    
    if algorithm == "ppo":
        print(f"🤖 Creating PPO model...")
        return PPO(
            "MlpPolicy",
            env,
            **PPO_CONFIG,
            device="auto",
        )
    
    elif algorithm == "sac":
        print(f"🤖 Creating SAC model...")
        return SAC(
            "MlpPolicy",
            env,
            **SAC_CONFIG,
            device="auto",
        )
    
    elif algorithm == "td3":
        print(f"🤖 Creating TD3 model...")
        return TD3(
            "MlpPolicy",
            env,
            **TD3_CONFIG,
            device="auto",
        )
    
    else:
        raise ValueError(
            f"Unsupported algorithm: {algorithm}. "
            f"Choose from: ppo, sac, td3"
        )
