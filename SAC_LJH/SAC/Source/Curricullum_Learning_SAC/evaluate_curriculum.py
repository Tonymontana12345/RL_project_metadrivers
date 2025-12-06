"""
커리큘럼 러닝 모델 평가 스크립트
각 Stage 모델을 해당 맵에서 평가
"""

import os
import argparse
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv

from config import (
    STAGE1_ENV_CONFIG, STAGE2_ENV_CONFIG, STAGE3_ENV_CONFIG, STAGE4_ENV_CONFIG,
    STAGE5_ENV_CONFIG, STAGE6_ENV_CONFIG,
    STAGE1_TRAINING, STAGE2_TRAINING, STAGE3_TRAINING, STAGE4_TRAINING,
    STAGE5_TRAINING, STAGE6_TRAINING,
    FIXED_SEED, set_global_seed
)
from utils.path_utils import get_model_path
from envs.metadrive_env import make_env


def evaluate_model(model_path, env_config, n_episodes=10, seed=None):
    """
    모델을 특정 환경에서 평가

    Args:
        model_path: 모델 경로
        env_config: 환경 설정
        n_episodes: 평가 에피소드 수
        seed: 시드 (None이면 env_config의 시드 사용)

    Returns:
        dict: 평가 결과
    """
    # 시드 설정
    if seed is not None:
        set_global_seed(seed)

    # 환경 생성
    eval_config = env_config.copy()
    if seed is not None:
        eval_config["start_seed"] = seed

    env = DummyVecEnv([make_env(seed=eval_config.get("start_seed"), render=False, config=eval_config)])

    # 모델 로드
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"모델을 찾을 수 없습니다: {model_path}")

    model = SAC.load(model_path, env=env)

    # 평가
    episode_rewards = []
    episode_lengths = []
    success_count = 0
    crash_count = 0
    out_of_road_count = 0

    for episode in range(n_episodes):
        obs = env.reset()
        done = False
        total_reward = 0
        steps = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            total_reward += reward[0]
            steps += 1

        episode_rewards.append(total_reward)
        episode_lengths.append(steps)

        # 종료 상태 확인
        if info[0].get("arrive_dest", False):
            success_count += 1
        if info[0].get("crash", False):
            crash_count += 1
        if info[0].get("out_of_road", False):
            out_of_road_count += 1

    env.close()

    # 통계 계산
    results = {
        "mean_reward": np.mean(episode_rewards),
        "std_reward": np.std(episode_rewards),
        "min_reward": np.min(episode_rewards),
        "max_reward": np.max(episode_rewards),
        "mean_length": np.mean(episode_lengths),
        "success_rate": success_count / n_episodes,
        "crash_rate": crash_count / n_episodes,
        "out_of_road_rate": out_of_road_count / n_episodes,
    }

    return results


def evaluate_curriculum(stages=[1, 2, 3, 4, 5, 6], n_episodes=10, seed=FIXED_SEED):
    """
    커리큘럼 러닝 모델들을 평가

    Args:
        stages: 평가할 Stage 리스트
        n_episodes: 각 Stage당 평가 에피소드 수
        seed: 평가 시드
    """
    # Stage 설정 (6단계: O → OC → TOC → TOCS → TSCOX → TSCOXrR)
    stage_configs = {
        1: {
            "name": "Stage 1: 로터리만",
            "model_name": STAGE1_TRAINING["model_name"],
            "env_config": STAGE1_ENV_CONFIG,
            "map": "O",
            "target_success_rate": 0.7,
        },
        2: {
            "name": "Stage 2: 로터리 + 곡선",
            "model_name": STAGE2_TRAINING["model_name"],
            "env_config": STAGE2_ENV_CONFIG,
            "map": "OC",
            "target_success_rate": 0.6,
        },
        3: {
            "name": "Stage 3: T교차로 + 로터리 + 곡선",
            "model_name": STAGE3_TRAINING["model_name"],
            "env_config": STAGE3_ENV_CONFIG,
            "map": "TOC",
            "target_success_rate": 0.5,
        },
        4: {
            "name": "Stage 4: TOCS",
            "model_name": STAGE4_TRAINING["model_name"],
            "env_config": STAGE4_ENV_CONFIG,
            "map": "TOCS",
            "target_success_rate": 0.3,
        },
        5: {
            "name": "Stage 5: 십자교차로 추가 (TSCOX)",
            "model_name": STAGE5_TRAINING["model_name"],
            "env_config": STAGE5_ENV_CONFIG,
            "map": "TSCOX",
            "target_success_rate": 0.25,
        },
        6: {
            "name": "Stage 6: 램프 추가 (TSCOXrR)",
            "model_name": STAGE6_TRAINING["model_name"],
            "env_config": STAGE6_ENV_CONFIG,
            "map": "TSCOXrR",
            "target_success_rate": 0.2,
        },
    }

    print("\n" + "="*60)
    print("🎓 커리큘럼 러닝 모델 평가")
    print("="*60)
    print(f"평가 시드: {seed}")
    print(f"에피소드 수: {n_episodes}")
    print("="*60 + "\n")

    all_results = {}

    for stage_num in stages:
        config = stage_configs[stage_num]
        model_path = get_model_path(config["model_name"])

        print(f"\n{'='*60}")
        print(f"📊 {config['name']} (맵: {config['map']})")
        print(f"{'='*60}")
        print(f"모델: {model_path}")

        # 모델 존재 확인
        if not os.path.exists(model_path):
            print(f"❌ 모델을 찾을 수 없습니다. 먼저 학습을 진행하세요.")
            print(f"   python train_curriculum.py --start_stage {stage_num} --end_stage {stage_num}")
            continue

        # 평가 실행
        try:
            results = evaluate_model(
                model_path=model_path,
                env_config=config["env_config"],
                n_episodes=n_episodes,
                seed=seed
            )

            all_results[stage_num] = results

            # 결과 출력
            print(f"\n평균 보상: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
            print(f"보상 범위: [{results['min_reward']:.2f}, {results['max_reward']:.2f}]")
            print(f"평균 길이: {results['mean_length']:.1f} 스텝")
            print(f"성공률: {results['success_rate']*100:.1f}%")
            print(f"충돌률: {results['crash_rate']*100:.1f}%")
            print(f"이탈률: {results['out_of_road_rate']*100:.1f}%")

            # 목표 달성 여부
            target_rate = config["target_success_rate"]
            if results["success_rate"] >= target_rate:
                print(f"✅ 목표 성공률 달성! (목표: {target_rate*100:.0f}%)")
            else:
                print(f"⚠️  목표 미달성 (목표: {target_rate*100:.0f}%, 현재: {results['success_rate']*100:.1f}%)")
                print(f"💡 더 많은 학습이 필요할 수 있습니다.")

        except Exception as e:
            print(f"❌ 평가 실패: {e}")
            continue

    # 전체 요약
    if all_results:
        print("\n" + "="*60)
        print("📈 전체 요약")
        print("="*60)

        for stage_num, results in all_results.items():
            config = stage_configs[stage_num]
            status = "✅" if results["success_rate"] >= config["target_success_rate"] else "⚠️"
            print(f"{status} Stage {stage_num} ({config['map']}): "
                  f"성공률 {results['success_rate']*100:.1f}%, "
                  f"보상 {results['mean_reward']:.2f}")

        print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="커리큘럼 러닝 모델 평가")
    parser.add_argument("--stages", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6],
                       help="평가할 Stage 번호 (예: --stages 1 2 3 4 5 6)")
    parser.add_argument("--n-episodes", type=int, default=10,
                       help="각 Stage당 평가 에피소드 수")
    parser.add_argument("--seed", type=int, default=FIXED_SEED,
                       help="평가 시드")

    args = parser.parse_args()

    evaluate_curriculum(
        stages=args.stages,
        n_episodes=args.n_episodes,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
