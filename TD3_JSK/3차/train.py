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
    RANDOM_BLOCK_TRAINING,        # ✅ 랜덤 블록 학습 설정
    TRAIN_SEEDS,
    EXPERIMENTS,
    FIXED_SEED_ENV_CONFIG,
    MULTI_SEED_ENV_CONFIG,
    RANDOM_BLOCK_ENV_CONFIG,      # ✅ 랜덤 블록 환경 설정
    set_global_seed,              # 랜덤 시드 고정 함수
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
    eval_freq=25000,
    n_eval_episodes=20,
    env_config=None,
    verbose=True,
):
    """
    Main training function.
    """
    # 랜덤 시드 고정 (재현성 보장)
    set_global_seed(seed)

    # 환경 설정 결정
    if env_config is None:
        env_config = FIXED_SEED_ENV_CONFIG

    print("\n" + "=" * 60)
    print(f"🚗 PGDrive Training Started: {model_name}")
    print("=" * 60)
    print(f"Algorithm: {algorithm.upper()}")
    print(f"Seed: {seed} (고정됨 - 재현성 보장)")
    print(f"Num Scenarios: {env_config.get('num_scenarios', 1)}")
    print(f"Total Timesteps: {total_timesteps:,}")
    # 랜덤 블록 맵 사용 여부 안내
    if env_config.get("randomize_block_order", False):
        print(f"Map Mode: RANDOM BLOCK ORDER ({env_config.get('map_blocks', ['T','S','C','O'])}, "
              f"length={env_config.get('num_map_blocks', 4)})")
    else:
        print(f"Map Mode: FIXED MAP ({env_config.get('map', 'N/A')})")
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60 + "\n")

    # Create environment
    print("📦 Creating environment...")
    env = DummyVecEnv([make_env(seed=seed, render=False, config=env_config)])
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
        verbose=1,
    )

    # Note: EvalCallback은 MetaDrive 엔진 재초기화 문제로 인해 비활성화
    # 학습 완료 후 evaluate.py로 별도 평가 권장

    # Start training
    print("🎓 Starting training...\n")
    print("⚠️  Note: 학습 중 평가는 MetaDrive 제약으로 비활성화됨")
    print("    학습 완료 후 'python evaluate.py'로 평가하세요\n")
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

    print("\n" + "=" * 60)
    print("🎉 Training Complete!")
    print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an RL agent for MetaDrive.")

    # 실험 방식 선택: --experiment 또는 --mode
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--experiment",
        type=str,
        choices=list(EXPERIMENTS.keys()),
        help="Pre-defined experiment to run (e.g., exp1_fixed_seed)",
    )
    group.add_argument(
        "--mode",
        type=str,
        default="fixed",
        choices=["fixed", "quick", "multi", "random_blocks"],  # ✅ random_blocks 추가
        help="Training mode: fixed, quick, multi seed, or random_blocks",
    )

    parser.add_argument(
        "--algorithm",
        type=str,
        default="ppo",
        choices=["ppo", "sac", "td3"],
        help="RL algorithm to use: PPO, SAC, or TD3.",
    )
    args = parser.parse_args()

    # 모드별 설정 딕셔너리
    MODE_CONFIGS = {
        # 기존 TSCO 고정 맵
        "fixed": {
            "config": FIXED_SEED_TRAINING,
            "seed": FIXED_SEED,
            "env_config": FIXED_SEED_ENV_CONFIG,
        },
        # 빠른 테스트
        "quick": {
            "config": QUICK_TEST_TRAINING,
            "seed": FIXED_SEED,
            "env_config": FIXED_SEED_ENV_CONFIG,
        },
        # 다중 시드 환경 (여러 시나리오)
        "multi": {
            "config": MULTI_SEED_TRAINING,
            "seed": TRAIN_SEEDS[0],
            "env_config": MULTI_SEED_ENV_CONFIG,
        },
        # ✅ 랜덤 블록 맵 학습 모드
        #  - config.py의 RANDOM_BLOCK_TRAINING + RANDOM_BLOCK_ENV_CONFIG 사용
        "random_blocks": {
            "config": RANDOM_BLOCK_TRAINING,
            "seed": FIXED_SEED,  # 시드는 동일하게, 맵 구조만 에피소드마다 랜덤
            "env_config": RANDOM_BLOCK_ENV_CONFIG,
        },
    }

    # 실험 프로토콜 사용 또는 모드 사용
    if args.experiment:
        # EXPERIMENTS에서 설정 로드
        exp = EXPERIMENTS[args.experiment]
        config = exp["train_config"]
        seed = (
            exp["test_seeds"][0]
            if isinstance(exp["test_seeds"], list)
            else exp["test_seeds"]
        )
        env_config = exp.get("env_config", FIXED_SEED_ENV_CONFIG)

        print(f"\n🔬 실험 프로토콜: {args.experiment}")
        print(f"📝 설명: {exp['description']}\n")
    else:
        # 기존 --mode 방식
        mode_data = MODE_CONFIGS[args.mode]
        config = mode_data["config"]
        seed = mode_data["seed"]
        env_config = mode_data["env_config"]

        print(f"\n🧪 Training mode: {args.mode}")
        if args.mode == "random_blocks":
            print("   → 에피소드마다 T/S/C/O 블록 순서가 랜덤인 맵에서 학습합니다.\n")

    # 모델명에 알고리즘 반영
    #  - 예: ppo_tsco_random_map_500k → td3_tsco_random_map_500k
    model_name = config["model_name"].replace("ppo", args.algorithm)

    train(
        total_timesteps=config["total_timesteps"],
        save_freq=config["save_freq"],
        model_name=model_name,
        seed=seed,
        algorithm=args.algorithm,
        env_config=env_config,
    )
