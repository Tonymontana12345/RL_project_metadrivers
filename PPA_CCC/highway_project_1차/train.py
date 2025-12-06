"""
Main training script for the PGDrive project.

This script integrates all modules to run training experiments.
"""

import os
import argparse
from datetime import datetime

from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from config import (
    FIXED_SEED, 
    FIXED_SEED_TRAINING, 
    QUICK_TEST_TRAINING,
    MULTI_SEED_TRAINING,
    TRAIN_SEEDS,
    EXPERIMENTS
)
from utils.path_utils import get_model_path, get_log_path
from envs.metadrive_env import make_env
from agents.rl_agent import create_model

def train(
    total_timesteps,
    save_freq,
    model_name,
    seed,
    algorithm="ppo",
    eval_freq=None, # Not used for now
    n_eval_episodes=None, # Not used for now
    verbose=True
):
    """
    Main training function.
    """
    print("\n" + "="*60)
    print(f"🚗 PGDrive Training Started: {model_name}")
    print("="*60)
    print(f"Algorithm: {algorithm.upper()}")
    print(f"Seed: {seed}")
    print(f"Total Timesteps: {total_timesteps:,}")
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60 + "\n")

    # Create environment
    print("📦 Creating environment...")
    env = DummyVecEnv([make_env(seed=seed, render=False)])
    print("✅ Environment created.\n")

    # Create model
    model = create_model(env, algorithm=algorithm)
    print("✅ Model created.\n")

    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq,
        save_path=os.path.dirname(get_model_path(model_name)),
        name_prefix=model_name,
        save_replay_buffer=False,
        save_vecnormalize=True,
        verbose=1
    )

    # Start training
    print("🎓 Starting training...\n")
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=[checkpoint_callback],
            progress_bar=True,
        )
        print("\n✅ Training finished!")

    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted (Ctrl+C).")

    # Save final model
    final_model_path = get_model_path(model_name)
    model.save(final_model_path)
    print(f"\n💾 Final model saved: {final_model_path}")

    # Close environment
    env.close()

    print("\n" + "="*60)
    print("🎉 Training Complete!")
    print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an RL agent for MetaDrive.")
    
    # 실험 방식 선택: --experiment 또는 --mode
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--experiment", type=str, 
                      choices=list(EXPERIMENTS.keys()),
                      help="Pre-defined experiment to run (e.g., exp1_fixed_seed)")
    group.add_argument("--mode", type=str, default="fixed",
                      choices=["fixed", "quick", "multi"],
                      help="Training mode: fixed, quick, or multi seed")
    
    parser.add_argument("--algorithm", type=str, default="ppo", 
                       choices=["ppo", "sac", "td3"],
                       help="RL algorithm to use: PPO, SAC, or TD3.")
    args = parser.parse_args()
    
    # 모드별 설정 딕셔너리
    MODE_CONFIGS = {
        "fixed": {"config": FIXED_SEED_TRAINING, "seed": FIXED_SEED},
        "quick": {"config": QUICK_TEST_TRAINING, "seed": FIXED_SEED},
        "multi": {"config": MULTI_SEED_TRAINING, "seed": TRAIN_SEEDS[0]},
    }
    
    # 실험 프로토콜 사용 또는 모드 사용
    if args.experiment:
        # EXPERIMENTS에서 설정 로드
        exp = EXPERIMENTS[args.experiment]
        config = exp["train_config"]
        seed = exp["test_seeds"][0] if isinstance(exp["test_seeds"], list) else exp["test_seeds"]
        
        print(f"\n🔬 실험 프로토콜: {args.experiment}")
        print(f"📝 설명: {exp['description']}\n")
    else:
        # 기존 --mode 방식
        mode_data = MODE_CONFIGS[args.mode]
        config = mode_data["config"]
        seed = mode_data["seed"]
    
    # 모델명에 알고리즘 반영
    model_name = config["model_name"].replace("ppo", args.algorithm)
    
    train(
        total_timesteps=config["total_timesteps"],
        save_freq=config["save_freq"],
        model_name=model_name,
        seed=seed,
        algorithm=args.algorithm
    )
