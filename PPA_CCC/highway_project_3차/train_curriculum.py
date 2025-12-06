"""
커리큘럼 러닝 학습 스크립트

목표: 충돌 없이 안전하게 완주
방법: 쉬운 환경 → 중간 환경 → 어려운 환경 (3단계 점진적 학습)

실행 방법:
    python train_curriculum.py
"""

import os
import argparse
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

# 커리큘럼 전용 설정 import
from config_curriculum import (
    set_global_seed,
    FIXED_SEED,
    TRAIN_SEEDS,
    TEST_SEEDS,
    BASE_ENV_CONFIG,
    CURRICULUM_CONFIG,
    PPO_CONFIG,
    CURRICULUM_TRAINING,
    MODELS_DIR,
    LOGS_DIR,
    print_curriculum_info,
)

# 환경 생성 함수 import
from envs.metadrive_env import make_env


def create_curriculum_env(stage_config, train_seeds):
    """
    커리큘럼 단계에 맞는 환경 생성
    
    Args:
        stage_config: 단계별 설정 dict
        train_seeds: 학습 시드 리스트
    
    Returns:
        DummyVecEnv: 벡터화된 환경
    """
    # 기본 환경 설정 복사
    env_config = BASE_ENV_CONFIG.copy()
    
    # 단계별 설정 적용
    env_config["traffic_density"] = stage_config["traffic_density"]
    env_config["horizon"] = stage_config["horizon"]
    
    # 다중 시드 학습을 위한 설정
    env_config["start_seed"] = train_seeds[0]
    env_config["num_scenarios"] = len(train_seeds)
    
    # 벡터화 (단일 환경, make_env가 callable을 반환)
    vec_env = DummyVecEnv([make_env(seed=None, render=False, config=env_config)])
    
    return vec_env


def train_curriculum():
    """
    커리큘럼 러닝으로 학습
    """
    # 설정 정보 출력
    print_curriculum_info()
    
    # 랜덤 시드 고정
    set_global_seed(FIXED_SEED)
    
    print("\n" + "="*70)
    print("🚀 커리큘럼 러닝 학습 시작!")
    print("="*70)
    print(f"시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"학습 시드: {TRAIN_SEEDS}")
    print(f"테스트 시드: {TEST_SEEDS}")
    print(f"모델 이름: {CURRICULUM_TRAINING['model_name']}")
    print("="*70 + "\n")
    
    # 변수 초기화
    model = None
    total_steps_trained = 0
    
    # 3단계 커리큘럼 학습
    for stage_num, stage_name in enumerate(["stage_1", "stage_2", "stage_3"], 1):
        stage_config = CURRICULUM_CONFIG[stage_name]
        
        print("\n" + "="*70)
        print(f"📚 단계 {stage_num}/3: {stage_config['name']} 환경")
        print("="*70)
        print(f"🎯 목표: {stage_config['description']}")
        print(f"📊 Traffic Density: {stage_config['traffic_density']}")
        print(f"⏱️  Horizon: {stage_config['horizon']}")
        print(f"🔢 학습 스텝: {stage_config['steps']:,}")
        print(f"💡 예상 성능: {stage_config['expected_success_rate']}")
        print("="*70 + "\n")
        
        # 환경 생성
        print("🌍 환경 생성 중...")
        env = create_curriculum_env(stage_config, TRAIN_SEEDS)
        print(f"✅ 환경 생성 완료 ({len(TRAIN_SEEDS)}개 시드: {TRAIN_SEEDS})")
        
        # 모델 생성 또는 로드
        if model is None:
            # 첫 단계: 새 모델 생성
            print("\n🆕 새 PPO 모델 생성...")
            log_dir = os.path.join(LOGS_DIR, CURRICULUM_TRAINING['model_name'])
            os.makedirs(log_dir, exist_ok=True)
            
            model = PPO(
                "MlpPolicy",
                env,
                **PPO_CONFIG,
            )
            print("✅ 모델 생성 완료")
        else:
            # 다음 단계: 기존 모델에 새 환경 적용
            print(f"\n🔄 이전 단계({stage_num-1})의 모델을 로드하여 이어서 학습...")
            model.set_env(env)
            print("✅ 환경 업데이트 완료")
        
        # 콜백 설정
        checkpoint_dir = os.path.join(MODELS_DIR, f"{CURRICULUM_TRAINING['model_name']}_checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint_callback = CheckpointCallback(
            save_freq=CURRICULUM_TRAINING['save_freq'],
            save_path=checkpoint_dir,
            name_prefix=f"{stage_name}",
            save_replay_buffer=False,
            save_vecnormalize=True,
        )
        
        # 학습 시작
        print(f"\n🚀 단계 {stage_num} 학습 시작!")
        print(f"⏱️  예상 소요 시간: 약 1시간")
        print("-"*70 + "\n")
        
        try:
            model.learn(
                total_timesteps=stage_config["steps"],
                callback=checkpoint_callback,
                progress_bar=True,
                reset_num_timesteps=False,  # 이전 단계 스텝 카운트 유지
            )
        except KeyboardInterrupt:
            print("\n\n⚠️  사용자가 학습을 중단했습니다.")
            print("현재까지의 모델을 저장합니다...\n")
        
        total_steps_trained += stage_config["steps"]
        
        # 단계별 모델 저장
        stage_model_path = os.path.join(
            MODELS_DIR,
            f"{CURRICULUM_TRAINING['model_name']}_{stage_name}_{total_steps_trained}.zip"
        )
        model.save(stage_model_path)
        
        print("\n" + "="*70)
        print(f"💾 단계 {stage_num} 완료! 모델 저장됨")
        print("="*70)
        print(f"저장 경로: {stage_model_path}")
        print(f"누적 학습 스텝: {total_steps_trained:,}")
        print("="*70 + "\n")
        
        # 환경 정리
        env.close()
    
    # 최종 모델 저장
    final_model_path = os.path.join(
        MODELS_DIR,
        f"{CURRICULUM_TRAINING['model_name']}.zip"
    )
    model.save(final_model_path)
    
    print("\n" + "="*70)
    print("🎉 커리큘럼 러닝 학습 완료!")
    print("="*70)
    print(f"종료 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\n최종 모델 저장 경로:")
    print(f"  {final_model_path}")
    print(f"\n총 학습 스텝: {total_steps_trained:,}")
    print(f"\n저장된 모델 목록:")
    print(f"  1. Stage 1 (300k): {CURRICULUM_TRAINING['model_name']}_stage_1_300000.zip")
    print(f"  2. Stage 2 (600k): {CURRICULUM_TRAINING['model_name']}_stage_2_600000.zip")
    print(f"  3. Stage 3 (900k): {CURRICULUM_TRAINING['model_name']}_stage_3_900000.zip")
    print(f"  4. Final: {CURRICULUM_TRAINING['model_name']}.zip")
    print(f"\n다음 단계:")
    print(f"  평가: python evaluate.py --model {final_model_path}")
    print("="*70 + "\n")


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="커리큘럼 러닝 학습")
    parser.add_argument(
        "--info",
        action="store_true",
        help="커리큘럼 설정 정보만 출력하고 종료"
    )
    
    args = parser.parse_args()
    
    if args.info:
        # 설정 정보만 출력
        print_curriculum_info()
        return
    
    # 커리큘럼 학습 실행
    train_curriculum()


if __name__ == "__main__":
    main()

