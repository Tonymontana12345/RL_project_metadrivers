"""
TD3 학습 스크립트 (자율주행 최적화)

TD3 알고리즘을 사용하여 MetaDrive 환경에서 자율주행을 학습합니다.
"""

import os
import argparse
import signal
import sys
from datetime import datetime

from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from config import (
    FIXED_SEED, 
    FIXED_SEED_TRAINING, 
    QUICK_TEST_TRAINING,
    MULTI_SEED_TRAINING,
    TRAIN_SEEDS,
    EXPERIMENTS,
    FIXED_SEED_ENV_CONFIG,
    MULTI_SEED_ENV_CONFIG,
    set_global_seed
)
from utils.path_utils import get_model_path, get_log_path
from envs.metadrive_env import make_env
from agents.rl_agent import create_model


# 전역 변수 (중단 신호 처리용)
interrupted = False

def signal_handler(sig, frame):
    """중단 신호 처리"""
    global interrupted
    interrupted = True
    print("\n\n⚠️  중단 신호 수신! 현재까지 학습된 모델을 저장합니다...")
    sys.exit(0)


def train(
    total_timesteps,
    save_freq,
    model_name,
    seed,
    algorithm="td3",
    eval_freq=25000,
    n_eval_episodes=10,
    env_config=None,
    verbose=True
):
    """
    TD3 학습 함수 (자율주행 최적화)
    
    Args:
        total_timesteps: 총 학습 스텝 수
        save_freq: 모델 저장 빈도
        model_name: 모델 이름
        seed: 랜덤 시드
        algorithm: 알고리즘 (기본값: "td3")
        eval_freq: 평가 빈도
        n_eval_episodes: 평가 에피소드 수
        env_config: 환경 설정 dict
        verbose: 출력 여부
    """
    # 중단 신호 핸들러 등록
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # 랜덤 시드 고정 (재현성 보장)
    set_global_seed(seed)
    
    # 환경 설정 결정
    if env_config is None:
        env_config = FIXED_SEED_ENV_CONFIG
    
    print("\n" + "="*60)
    print(f"🚗 TD3 자율주행 학습 시작: {model_name}")
    print("="*60)
    print(f"Algorithm: {algorithm.upper()}")
    print(f"Seed: {seed} (고정됨 - 재현성 보장)")
    print(f"Num Scenarios: {env_config.get('num_scenarios', 1)}")
    print(f"Total Timesteps: {total_timesteps:,}")
    print(f"Save Frequency: {save_freq:,} steps")
    print(f"Eval Frequency: {eval_freq:,} steps")
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60 + "\n")
    
    # 환경 생성
    print("📦 환경 생성 중...")
    try:
        env_fn = make_env(seed=seed, render=False, config=env_config)
        env = DummyVecEnv([env_fn])
        
        # 액션/관측 공간 확인
        print(f"✅ 환경 생성 완료")
        print(f"   관측 공간: {env.observation_space}")
        print(f"   액션 공간: {env.action_space}")
        
        # 연속 액션 공간 체크
        if not hasattr(env.action_space, 'low') or not hasattr(env.action_space, 'high'):
            raise ValueError(
                f"❌ TD3는 연속 액션 공간을 요구합니다. "
                f"현재 액션 공간: {env.action_space}"
            )
    except Exception as e:
        print(f"❌ 환경 생성 실패: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print()
    
    # 모델 생성
    print("🤖 TD3 모델 생성 중...")
    try:
        model = create_model(env, algorithm=algorithm)
        print("✅ 모델 생성 완료!\n")
    except Exception as e:
        print(f"❌ 모델 생성 실패: {e}")
        import traceback
        traceback.print_exc()
        env.close()
        return
    
    # 콜백 설정
    model_dir = os.path.dirname(get_model_path(model_name))
    os.makedirs(model_dir, exist_ok=True)
    
    # 체크포인트 콜백
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq,
        save_path=model_dir,
        name_prefix=model_name,
        save_replay_buffer=False,
        save_vecnormalize=True,
        verbose=1
    )
    
    # 평가 콜백은 MetaDrive 환경에서 문제가 발생할 수 있으므로 비활성화
    # 대신 주기적으로 수동 저장만 수행
    print("⚠️  평가 콜백 비활성화 (MetaDrive 호환성 문제 방지)")
    print("   체크포인트 콜백만 사용합니다.\n")
    callbacks = [checkpoint_callback]
    
    # 학습 시작
    print("🎓 학습 시작...\n")
    print("💡 팁:")
    print("   - TD3는 오프폴리시 알고리즘이므로 초기에는 탐험이 많이 필요합니다")
    print("   - 학습이 안정화되기까지 시간이 걸릴 수 있습니다")
    print("   - TensorBoard에서 학습 진행 상황을 확인하세요")
    print("   - 모델은 주기적으로 저장됩니다")
    print("   - Ctrl+C로 안전하게 중단할 수 있습니다\n")
    
    try:
        # 학습 진행 상황 로깅 강화
        print(f"📊 학습 진행 중... (목표: {total_timesteps:,} 스텝)")
        print("   진행 상황은 TensorBoard에서 확인하세요\n")
        
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            progress_bar=True,
            log_interval=10,  # 10 스텝마다 로그 출력
            tb_log_name=f"{model_name}_training",
        )
        print("\n✅ 학습 완료!")
        
    except KeyboardInterrupt:
        print("\n⚠️  학습 중단 (Ctrl+C)")
        print("   현재까지 학습된 모델을 저장합니다...")
        interrupted = True
        
    except Exception as e:
        print(f"\n❌ 학습 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        print("\n   현재까지 학습된 모델을 저장합니다...")
        interrupted = True
    
    finally:
        # 최종 모델 저장 (항상 실행)
        try:
            final_model_path = get_model_path(model_name)
            model.save(final_model_path)
            print(f"\n💾 최종 모델 저장: {final_model_path}")
        except Exception as e:
            print(f"\n⚠️  모델 저장 실패: {e}")
        
        # 환경 종료
        try:
            env.close()
        except:
            pass
        
        if interrupted:
            print("\n" + "="*60)
            print("⚠️  학습 중단됨")
            print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("="*60 + "\n")
        else:
            print("\n" + "="*60)
            print("🎉 학습 완료!")
            print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("="*60 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="TD3 알고리즘으로 MetaDrive 자율주행 학습"
    )
    
    # 실험 방식 선택
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--experiment", 
        type=str, 
        choices=list(EXPERIMENTS.keys()),
        help="사전 정의된 실험 실행 (예: exp1_fixed_seed)"
    )
    group.add_argument(
        "--mode", 
        type=str, 
        default="fixed",
        choices=["fixed", "quick", "multi"],
        help="학습 모드: fixed, quick, multi seed"
    )
    
    parser.add_argument(
        "--algorithm", 
        type=str, 
        default="td3", 
        choices=["td3"],
        help="RL 알고리즘 (현재 TD3만 지원)"
    )
    
    args = parser.parse_args()
    
    # 모드별 설정 딕셔너리
    MODE_CONFIGS = {
        "fixed": {
            "config": FIXED_SEED_TRAINING, 
            "seed": FIXED_SEED, 
            "env_config": FIXED_SEED_ENV_CONFIG
        },
        "quick": {
            "config": QUICK_TEST_TRAINING, 
            "seed": FIXED_SEED, 
            "env_config": FIXED_SEED_ENV_CONFIG
        },
        "multi": {
            "config": MULTI_SEED_TRAINING, 
            "seed": TRAIN_SEEDS[0], 
            "env_config": MULTI_SEED_ENV_CONFIG
        },
    }
    
    # 실험 프로토콜 사용 또는 모드 사용
    if args.experiment:
        # EXPERIMENTS에서 설정 로드
        exp = EXPERIMENTS[args.experiment]
        config = exp["train_config"]
        seed = exp["test_seeds"][0] if isinstance(exp["test_seeds"], list) else exp["test_seeds"]
        env_config = exp.get("env_config", FIXED_SEED_ENV_CONFIG)
        
        print(f"\n🔬 실험 프로토콜: {args.experiment}")
        print(f"📝 설명: {exp['description']}\n")
    else:
        # 기존 --mode 방식
        mode_data = MODE_CONFIGS[args.mode]
        config = mode_data["config"]
        seed = mode_data["seed"]
        env_config = mode_data["env_config"]
    
    # 학습 실행
    train(
        total_timesteps=config["total_timesteps"],
        save_freq=config["save_freq"],
        model_name=config["model_name"],
        seed=seed,
        algorithm=args.algorithm,
        eval_freq=config.get("eval_freq", 25000),
        n_eval_episodes=config.get("n_eval_episodes", 10),
        env_config=env_config
    )
