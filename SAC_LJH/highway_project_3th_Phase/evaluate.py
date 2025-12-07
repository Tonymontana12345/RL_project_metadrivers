"""
모델 평가 스크립트

학습된 모델을 다양한 시드에서 평가하여 일반화 성능을 측정합니다.
"""

import gymnasium as gym
from stable_baselines3 import PPO, SAC, TD3
import numpy as np
import pandas as pd
from tqdm import tqdm
import json
import os

from config import (
    FIXED_SEED,
    TEST_SEEDS,
    FIXED_SEED_ENV_CONFIG,
    EVALUATION_CONFIG,
)
from utils.path_utils import get_result_path
from envs.metadrive_env import make_env


def detect_algorithm(model_path):
    """
    모델 파일명에서 알고리즘 자동 감지
    
    Args:
        model_path: 모델 파일 경로
    
    Returns:
        str: 알고리즘 이름 ('ppo', 'sac', 'td3')
    """
    model_name = os.path.basename(model_path).lower()
    
    if 'sac' in model_name:
        return 'sac'
    elif 'td3' in model_name:
        return 'td3'
    elif 'ppo' in model_name:
        return 'ppo'
    else:
        # 기본값은 PPO
        return 'ppo'


def load_model(model_path, algorithm=None):
    """
    알고리즘에 맞는 모델 로드
    
    Args:
        model_path: 모델 파일 경로
        algorithm: 알고리즘 이름 (None이면 자동 감지)
    
    Returns:
        model: 로드된 모델
    
    Raises:
        ValueError: 지원하지 않는 알고리즘
    """
    if algorithm is None:
        algorithm = detect_algorithm(model_path)
    
    algorithm = algorithm.lower()
    
    if algorithm == 'ppo':
        return PPO.load(model_path)
    elif algorithm == 'sac':
        return SAC.load(model_path)
    elif algorithm == 'td3':
        return TD3.load(model_path)
    else:
        raise ValueError(
            f"Unsupported algorithm: {algorithm}. "
            f"Choose from: ppo, sac, td3"
        )


def evaluate_model(model_path, test_seeds, n_episodes=20, render=False, verbose=True, env_config=None):
    """
    모델을 여러 시드에서 평가
    
    Args:
        model_path: 모델 파일 경로
        test_seeds: 테스트할 시드 리스트
        n_episodes: 각 시드당 에피소드 수
        render: 렌더링 여부
        verbose: 상세 출력 여부
        env_config: 환경 설정 (None이면 FIXED_SEED_ENV_CONFIG 사용)
    
    Returns:
        dict: 평가 결과
    """
    if verbose:
        print("\n" + "="*60)
        print("📊 모델 평가 시작")
        print("="*60)
        print(f"모델: {model_path}")
        print(f"테스트 시드: {test_seeds}")
        print(f"에피소드/시드: {n_episodes}")
        if env_config and env_config.get("map"):
            print(f"맵 설정: {env_config.get('map')}")
        print("="*60 + "\n")
    
    # 모델 로드 (알고리즘 자동 감지)
    try:
        algorithm = detect_algorithm(model_path)
        if verbose:
            print(f"🔍 감지된 알고리즘: {algorithm.upper()}")
        
        model = load_model(model_path, algorithm)
        if verbose:
            print("✅ 모델 로드 완료\n")
    except Exception as e:
        print(f"❌ 모델 로드 실패: {e}")
        return None
    
    # 결과 저장
    results = {
        "model_path": model_path,
        "test_seeds": test_seeds,
        "n_episodes": n_episodes,
        "seed_results": {},
        "summary": {}
    }
    
    # 각 시드별 평가
    for seed in test_seeds:
        if verbose:
            print(f"\n🎯 시드 {seed} 평가 중...")
        
        # 환경 생성 (make_env 사용, env_config 전달)
        env = make_env(seed=seed, render=render, config=env_config)()
        
        # 에피소드별 결과
        episode_rewards = []
        episode_lengths = []
        success_count = 0
        crash_count = 0
        out_of_road_count = 0
        
        # 진행 바
        iterator = tqdm(range(n_episodes), desc=f"Seed {seed}") if verbose else range(n_episodes)
        
        for episode in iterator:
            obs, info = env.reset()
            done = False
            total_reward = 0
            steps = 0
            
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                total_reward += reward
                steps += 1
            
            # 결과 기록
            episode_rewards.append(total_reward)
            episode_lengths.append(steps)
            
            if info.get("arrive_dest", False):
                success_count += 1
            if info.get("crash", False):
                crash_count += 1
            if info.get("out_of_road", False):
                out_of_road_count += 1
        
        env.close()
        
        # 시드별 통계
        seed_stats = {
            "mean_reward": np.mean(episode_rewards),
            "std_reward": np.std(episode_rewards),
            "min_reward": np.min(episode_rewards),
            "max_reward": np.max(episode_rewards),
            "mean_length": np.mean(episode_lengths),
            "success_rate": success_count / n_episodes,
            "crash_rate": crash_count / n_episodes,
            "out_of_road_rate": out_of_road_count / n_episodes,
            "episode_rewards": episode_rewards,
            "episode_lengths": episode_lengths,
        }
        
        results["seed_results"][seed] = seed_stats
        
        if verbose:
            print(f"\n  평균 보상: {seed_stats['mean_reward']:.2f} ± {seed_stats['std_reward']:.2f}")
            print(f"  성공률: {seed_stats['success_rate']*100:.1f}%")
            print(f"  충돌률: {seed_stats['crash_rate']*100:.1f}%")
    
    # 전체 요약 통계
    all_rewards = []
    all_success_rates = []
    
    for seed_stats in results["seed_results"].values():
        all_rewards.extend(seed_stats["episode_rewards"])
        all_success_rates.append(seed_stats["success_rate"])
    
    results["summary"] = {
        "overall_mean_reward": np.mean(all_rewards),
        "overall_std_reward": np.std(all_rewards),
        "overall_success_rate": np.mean(all_success_rates),
        "success_rate_std": np.std(all_success_rates),
    }
    
    if verbose:
        print("\n" + "="*60)
        print("📈 전체 요약")
        print("="*60)
        print(f"전체 평균 보상: {results['summary']['overall_mean_reward']:.2f} ± {results['summary']['overall_std_reward']:.2f}")
        print(f"전체 평균 성공률: {results['summary']['overall_success_rate']*100:.1f}% ± {results['summary']['success_rate_std']*100:.1f}%")
        
        # 성공률이 0%일 때 경고 메시지
        if results['summary']['overall_success_rate'] == 0:
            print("\n⚠️  경고: 성공률이 0%입니다!")
            print("💡 가능한 원인:")
            print("   1. 학습이 충분하지 않음 (더 많은 스텝 필요)")
            print("   2. 모델이 제대로 학습되지 않음")
            print("   3. 에이전트가 항상 도로를 벗어남")
            print("\n💡 해결 방법:")
            print("   1. TensorBoard로 학습 곡선 확인:")
            print("      tensorboard --logdir logs/")
            print("   2. 더 긴 학습 실행:")
            print("      python train.py --mode fixed  # 100K 스텝")
            print("   3. 디버깅 스크립트 실행:")
            print("      python debug_results.py --results results/evaluation_results.json")
        
        print("="*60 + "\n")
    
    return results


def compare_seeds(results, train_seed=FIXED_SEED):
    """
    학습 시드와 테스트 시드 성능 비교
    
    Args:
        results: evaluate_model 결과
        train_seed: 학습에 사용한 시드
    """
    print("\n" + "="*60)
    print("🔍 시드별 성능 비교")
    print("="*60)
    
    # 학습 시드 성능
    if train_seed in results["seed_results"]:
        train_stats = results["seed_results"][train_seed]
        print(f"\n📚 학습 시드 ({train_seed}):")
        print(f"  평균 보상: {train_stats['mean_reward']:.2f}")
        print(f"  성공률: {train_stats['success_rate']*100:.1f}%")
    
    # 테스트 시드 성능
    test_seeds = [s for s in results["seed_results"].keys() if s != train_seed]
    if test_seeds:
        print(f"\n🧪 테스트 시드 ({len(test_seeds)}개):")
        
        test_rewards = []
        test_success_rates = []
        
        for seed in test_seeds:
            stats = results["seed_results"][seed]
            test_rewards.append(stats["mean_reward"])
            test_success_rates.append(stats["success_rate"])
            print(f"  시드 {seed}: 보상 {stats['mean_reward']:.2f}, 성공률 {stats['success_rate']*100:.1f}%")
        
        print(f"\n  평균 보상: {np.mean(test_rewards):.2f} ± {np.std(test_rewards):.2f}")
        print(f"  평균 성공률: {np.mean(test_success_rates)*100:.1f}% ± {np.std(test_success_rates)*100:.1f}%")
        
        # 일반화 성능 (학습 시드 대비)
        if train_seed in results["seed_results"]:
            generalization_gap = train_stats["mean_reward"] - np.mean(test_rewards)
            print(f"\n📉 일반화 갭: {generalization_gap:.2f} (학습 시드 - 테스트 평균)")
            
            if generalization_gap > 5:
                print("  ⚠️  과적합 가능성 있음")
            elif generalization_gap < -2:
                print("  ✅ 일반화 성능 우수")
            else:
                print("  ✅ 적절한 일반화")
    
    print("="*60 + "\n")


def save_results(results, filename="evaluation_results.json"):
    """
    평가 결과 저장
    
    Args:
        results: 평가 결과
        filename: 저장 파일명
    """
    filepath = get_result_path(filename)
    
    # numpy 배열을 리스트로 변환
    results_json = results.copy()
    for seed, stats in results_json["seed_results"].items():
        stats["episode_rewards"] = [float(r) for r in stats["episode_rewards"]]
        stats["episode_lengths"] = [int(l) for l in stats["episode_lengths"]]
    
    with open(filepath, 'w') as f:
        json.dump(results_json, f, indent=2)
    
    print(f"💾 결과 저장: {filepath}")


def create_summary_table(results):
    """
    결과를 표로 정리
    
    Args:
        results: 평가 결과
    
    Returns:
        pd.DataFrame
    """
    data = []
    
    for seed, stats in results["seed_results"].items():
        data.append({
            "Seed": seed,
            "Mean Reward": f"{stats['mean_reward']:.2f}",
            "Std Reward": f"{stats['std_reward']:.2f}",
            "Success Rate": f"{stats['success_rate']*100:.1f}%",
            "Crash Rate": f"{stats['crash_rate']*100:.1f}%",
            "Mean Length": f"{stats['mean_length']:.1f}",
        })
    
    df = pd.DataFrame(data)
    return df


if __name__ == "__main__":
    """
    메인 실행
    
    사용법:
        # 기본 평가
        python evaluate.py
        
        # 특정 모델 평가
        python evaluate.py --model models/ppo_fixed_seed_1000.zip
        
        # 렌더링과 함께 평가
        python evaluate.py --render
    """
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description="PGDrive 모델 평가")
    parser.add_argument("--model", type=str, default="models/ppo_fixed_seed_1000.zip",
                       help="평가할 모델 경로")
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                       help="테스트 시드 (기본: config의 TEST_SEEDS)")
    parser.add_argument("--episodes", type=int, default=20,
                       help="각 시드당 에피소드 수")
    parser.add_argument("--render", action="store_true",
                       help="렌더링 활성화")
    parser.add_argument("--save", type=str, default="evaluation_results.json",
                       help="결과 저장 파일명")
    parser.add_argument("--map", type=str, default=None,
                       help="맵 블록 타입 (예: TSCO, TRSCO 등, None이면 config 사용)")
    


    parser.add_argument("--record-gif", action="store_true",
                       help="실패한 시드를 GIF로 녹화 (record_driving_gif.py 호출)")
    parser.add_argument("--gif-threshold", type=float, default=0.3,
                      help="GIF 녹화 임계값 (성공률이 이 값 이하면 녹화, 기본: 0.3)")
    parser.add_argument("--record-success", action="store_true",
                      help="성공한 시드를 GIF로 녹화 (성공률 70% 이상)")
    
    
    args = parser.parse_args()
    
    # 테스트 시드 설정
    test_seeds = args.seeds if args.seeds else TEST_SEEDS
    
    # 환경 설정 (맵 블록 타입 지정)
    eval_env_config = None
    if args.map:
        eval_env_config = FIXED_SEED_ENV_CONFIG.copy()
        eval_env_config["map"] = args.map
        print(f"\n🗺️  맵 블록 타입: {args.map}")
    
    # 평가 실행
    results = evaluate_model(
        model_path=args.model,
        test_seeds=test_seeds,
        n_episodes=args.episodes,
        render=args.render,
        verbose=True,
        env_config=eval_env_config
    )
    
    if results:
        # 시드별 비교 (학습 시드는 TRAIN_SEEDS 사용)
        # compare_seeds(results, train_seed=FIXED_SEED)  # 더 이상 사용 안 함
        
        # 표 출력
        df = create_summary_table(results)
        print("\n📋 결과 표:")
        print(df.to_string(index=False))
        print()
        
        # 결과 저장
        save_results(results, args.save)


        if args.record_gif:
            import subprocess
            os.makedirs("gifs/failures", exist_ok=True)

            print("\n" + "="*60)
            print("🎬 실패 시드 GIF 녹화")
            print("="*60)
            
            # 실패한 시드 찾기
            failed_seeds = []
            for seed, stats in results["seed_results"].items():
                if stats["success_rate"] <= args.gif_threshold:
                    failed_seeds.append(seed)
            
            if failed_seeds:
                print(f"\n실패 시드 ({len(failed_seeds)}개): {failed_seeds}")
                print(f"임계값: {args.gif_threshold*100:.0f}% 이하\n")
                
                # evalute_with_record.py 호출
                for seed in failed_seeds:
                    print(f"\n📹 시드 {seed} 녹화 중...")
                    
                    cmd = [
                        "python", "evalute_with_record.py",
                        "--model", args.model,
                        "--seed", str(seed),
                        "--output", f"gifs/failures/seed_{seed}_failure.gif"
                    ]
                    
                    try:
                        subprocess.run(cmd, check=True)
                        print(f"✅ 완료: gifs/failures/seed_{seed}_failure.gif")
                    except subprocess.CalledProcessError as e:
                        print(f"❌ 실패: {e}")
                    except FileNotFoundError:
                        print("❌ evalute_with_record.py를 찾을 수 없습니다")
                        break
                
                print("\n" + "="*60)
                print(f"✅ GIF 녹화 완료: gifs/failures/ ({len(failed_seeds)}개)")
                print("="*60 + "\n")
            else:
                print(f"\n✅ 모든 시드가 임계값({args.gif_threshold*100:.0f}%)을 통과했습니다!")
                print("녹화할 실패 시드가 없습니다.\n")

# 성공 시드 녹화 (옵션)

        if args.record_success:

            print("\n" + "="*60)
            print("🎬 성공 시드 GIF 녹화")
            print("="*60)
            
            # 폴더 생성
            os.makedirs("gifs/success", exist_ok=True)            
            # 성공한 시드 찾기 (성공률 70% 이상)
            success_seeds = []
            for seed, stats in results["seed_results"].items():
                if stats["success_rate"] >= 0.7:
                    success_seeds.append(seed)
            
            if success_seeds:
                print(f"\n성공 시드 ({len(success_seeds)}개): {success_seeds}")
                print(f"임계값: 70% 이상\n")
                
                # record_driving_gif.py 호출
                for seed in success_seeds:
                    print(f"\n📹 시드 {seed} 녹화 중...")
                    
                    cmd = [
                        "python", "record_driving_gif.py",
                        "--model", args.model,
                        "--seed", str(seed),
                        "--output", f"gifs/success/seed_{seed}_success.gif"
                    ]
                    
                    try:
                        subprocess.run(cmd, check=True)
                        print(f"✅ 완료: gifs/success/seed_{seed}_success.gif")
                    except subprocess.CalledProcessError as e:
                        print(f"❌ 실패: {e}")
                    except FileNotFoundError:
                        print("❌ record_driving_gif.py를 찾을 수 없습니다")
                        break
                
                print("\n" + "="*60)
                print(f"✅ GIF 녹화 완료: gifs/success/ ({len(success_seeds)}개)")  
                print("="*60 + "\n")
            else:
                print(f"\n⚠️  성공률 70% 이상인 시드가 없습니다!")
                print("녹화할 성공 시드가 없습니다.\n")
            