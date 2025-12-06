"""
커리큘럼 러닝 학습 스크립트
단계적으로 난이도를 높여가며 학습
"""

import os
import argparse
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv

from config import (
    STAGE1_ENV_CONFIG, STAGE2_ENV_CONFIG, STAGE3_ENV_CONFIG, STAGE4_ENV_CONFIG,
    STAGE5_ENV_CONFIG, STAGE6_ENV_CONFIG,
    STAGE1_TRAINING, STAGE2_TRAINING, STAGE3_TRAINING, STAGE4_TRAINING,
    STAGE5_TRAINING, STAGE6_TRAINING,
    SAC_CONFIG, set_global_seed, FIXED_SEED
)
from utils.path_utils import get_model_path
from envs.metadrive_env import make_env

def train_stage(stage_num, training_config, env_config, prev_model_path=None):
    """
    특정 Stage 학습 (Off-Policy: Replay Buffer 저장/로딩)

    Args:
        stage_num: Stage 번호 (1-6)
        training_config: 학습 설정
        env_config: 환경 설정
        prev_model_path: 이전 Stage 모델 경로 (없으면 None)
    """
    print("\n" + "="*60)
    print(f"🎓 Stage {stage_num} 학습 시작")
    print(f"   맵: {env_config['map']}")
    print(f"   트래픽 밀도: {env_config['traffic_density']}")
    print(f"   학습 스텝: {training_config['total_timesteps']:,}")
    print("="*60 + "\n")

    # 랜덤 시드 고정
    set_global_seed(FIXED_SEED)

    # 환경 생성
    env = DummyVecEnv([make_env(seed=FIXED_SEED, render=False, config=env_config)])

    # 모델 생성 or 로드
    if prev_model_path and os.path.exists(prev_model_path):
        print(f"✅ 이전 모델 로드: {prev_model_path}")
        model = SAC.load(prev_model_path, env=env)

        # 🔥 Off-Policy 핵심: Replay Buffer 로드
        buffer_path = prev_model_path.replace(".zip", "_replay_buffer.pkl")
        if os.path.exists(buffer_path):
            print(f"📦 Replay Buffer 로드: {buffer_path}")
            model.load_replay_buffer(buffer_path)
            buffer_size = model.replay_buffer.size()
            print(f"   Buffer 크기: {buffer_size:,} transitions")

            # 디스크 절약: 이전 Buffer 파일 삭제
            os.remove(buffer_path)
            print(f"🗑️  이전 Buffer 삭제 (디스크 절약)")
        else:
            print(f"⚠️  Replay Buffer 파일 없음 (새로 시작)")
    else:
        print(f"✅ 새 모델 생성")
        model = SAC("MlpPolicy", env, **SAC_CONFIG)

    # 🔥 Off-Policy 핵심: reset_num_timesteps=False (Buffer 유지)
    model.learn(
        total_timesteps=training_config["total_timesteps"],
        reset_num_timesteps=False,  # Buffer 초기화 방지!
        progress_bar=True,
    )

    # 저장 (모델 + Replay Buffer)
    save_path = get_model_path(training_config["model_name"])
    model.save(save_path)
    print(f"\n💾 모델 저장: {save_path}")

    # 🔥 Off-Policy 핵심: Replay Buffer 저장
    buffer_path = save_path.replace(".zip", "_replay_buffer.pkl")
    model.save_replay_buffer(buffer_path)
    final_buffer_size = model.replay_buffer.size()
    buffer_size_mb = os.path.getsize(buffer_path) / (1024 * 1024)
    print(f"💾 Replay Buffer 저장: {buffer_path}")
    print(f"   Buffer 크기: {final_buffer_size:,} transitions ({buffer_size_mb:.1f} MB)")

    env.close()

    return save_path

def main():
    parser = argparse.ArgumentParser(description="커리큘럼 러닝 학습")
    parser.add_argument("--start_stage", type=int, default=1,
                       help="시작할 Stage (1~6)")
    parser.add_argument("--end_stage", type=int, default=6,
                       help="종료할 Stage (1~6)")
    args = parser.parse_args()

    # Stage 설정 (6단계: O → OC → TOC → TOCS → TSCOX → TSCOXrR)
    stages = [
        (1, STAGE1_TRAINING, STAGE1_ENV_CONFIG, None),
        (2, STAGE2_TRAINING, STAGE2_ENV_CONFIG, get_model_path("sac_stage1_roundabout")),
        (3, STAGE3_TRAINING, STAGE3_ENV_CONFIG, get_model_path("sac_stage2_roundabout_curve")),
        (4, STAGE4_TRAINING, STAGE4_ENV_CONFIG, get_model_path("sac_stage3_toc")),
        (5, STAGE5_TRAINING, STAGE5_ENV_CONFIG, get_model_path("sac_stage4_final")),
        (6, STAGE6_TRAINING, STAGE6_ENV_CONFIG, get_model_path("sac_stage5_intersection")),
    ]

    print("\n🚀 커리큘럼 러닝 시작! (6단계)")
    print("   Stage 1: O → Stage 2: OC → Stage 3: TOC")
    print("   Stage 4: TOCS → Stage 5: TSCOX → Stage 6: TSCOXrR")
    print(f"   실행 범위: Stage {args.start_stage} → Stage {args.end_stage}\n")

    # 순차 학습
    for stage_num, train_cfg, env_cfg, prev_model in stages:
        if args.start_stage <= stage_num <= args.end_stage:
            train_stage(stage_num, train_cfg, env_cfg, prev_model)

    print("\n" + "="*60)
    print("🎉 커리큘럼 러닝 완료!")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
