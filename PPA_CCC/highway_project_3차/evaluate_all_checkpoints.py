"""
모든 체크포인트를 평가하여 최적의 모델을 찾는 스크립트
과적합이 시작되기 전의 최적 시점을 찾습니다.
"""

import os
import glob
import json
import numpy as np
from tqdm import tqdm
import pandas as pd

from stable_baselines3 import PPO
from config import TEST_SEEDS, FIXED_SEED_ENV_CONFIG
from envs.metadrive_env import make_env
from stable_baselines3.common.vec_env import DummyVecEnv


def get_checkpoint_step(checkpoint_path):
    """체크포인트 파일명에서 스텝 수 추출"""
    basename = os.path.basename(checkpoint_path)
    if "steps" in basename:
        # ppo_multi_seed_50000_steps.zip -> 50000
        step_str = basename.split("_")[-2]
        return int(step_str)
    elif basename == "ppo_multi_seed.zip":
        # 최종 모델은 1,000,000 스텝
        return 1000000
    return 0


def evaluate_checkpoint(checkpoint_path, test_seeds, n_episodes=10):
    """
    단일 체크포인트를 평가합니다.
    
    Args:
        checkpoint_path: 체크포인트 경로
        test_seeds: 테스트할 시드 리스트
        n_episodes: 시드당 에피소드 수
    
    Returns:
        dict: 평가 결과 (평균 보상, 성공률, 충돌률 등)
    """
    print(f"\n📊 평가 중: {os.path.basename(checkpoint_path)}")
    
    # 모델 로드
    try:
        model = PPO.load(checkpoint_path)
    except Exception as e:
        print(f"❌ 모델 로드 실패: {e}")
        return None
    
    all_rewards = []
    all_success = []
    all_crashes = []
    all_lengths = []
    
    # 각 시드에서 평가
    for seed in test_seeds:
        env = DummyVecEnv([make_env(seed=seed, render=False, config=FIXED_SEED_ENV_CONFIG)])
        
        seed_rewards = []
        seed_success = 0
        seed_crashes = 0
        seed_lengths = []
        
        for _ in range(n_episodes):
            obs = env.reset()
            done = False
            episode_reward = 0
            episode_length = 0
            
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = env.step(action)
                episode_reward += reward[0]
                episode_length += 1
            
            seed_rewards.append(episode_reward)
            seed_lengths.append(episode_length)
            
            # 성공/충돌 확인
            if info[0].get("arrive_dest", False):
                seed_success += 1
            if info[0].get("crash", False) or info[0].get("crash_vehicle", False):
                seed_crashes += 1
        
        env.close()
        
        # 시드별 결과 수집
        all_rewards.extend(seed_rewards)
        all_success.append(seed_success / n_episodes * 100)
        all_crashes.append(seed_crashes / n_episodes * 100)
        all_lengths.extend(seed_lengths)
    
    # 전체 통계
    results = {
        "checkpoint": os.path.basename(checkpoint_path),
        "steps": get_checkpoint_step(checkpoint_path),
        "mean_reward": np.mean(all_rewards),
        "std_reward": np.std(all_rewards),
        "mean_success_rate": np.mean(all_success),
        "std_success_rate": np.std(all_success),
        "mean_crash_rate": np.mean(all_crashes),
        "mean_length": np.mean(all_lengths),
        "total_episodes": len(all_rewards)
    }
    
    print(f"  보상: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
    print(f"  성공률: {results['mean_success_rate']:.1f}%")
    print(f"  충돌률: {results['mean_crash_rate']:.1f}%")
    
    return results


def main():
    print("="*70)
    print("🔍 체크포인트 분석 - 과적합 최소화 모델 찾기")
    print("="*70)
    
    # 체크포인트 파일 찾기
    checkpoint_pattern = "models/ppo_multi_seed*.zip"
    checkpoints = glob.glob(checkpoint_pattern)
    
    if not checkpoints:
        print("❌ 체크포인트를 찾을 수 없습니다.")
        return
    
    # 스텝 수로 정렬
    checkpoints.sort(key=get_checkpoint_step)
    
    print(f"\n📁 발견된 체크포인트: {len(checkpoints)}개")
    print(f"📊 테스트 시드: {TEST_SEEDS}")
    print(f"🎯 에피소드/시드: 10개 (빠른 평가)")
    print(f"⏱️  예상 시간: 약 {len(checkpoints) * 1}분\n")
    
    # 모든 체크포인트 평가
    results = []
    for checkpoint in tqdm(checkpoints, desc="체크포인트 평가"):
        result = evaluate_checkpoint(checkpoint, TEST_SEEDS, n_episodes=10)
        if result:
            results.append(result)
    
    # 결과를 DataFrame으로 변환
    df = pd.DataFrame(results)
    df = df.sort_values("steps")
    
    print("\n" + "="*70)
    print("📈 체크포인트 분석 결과")
    print("="*70)
    print(df.to_string(index=False))
    
    # 최적 모델 찾기
    print("\n" + "="*70)
    print("🏆 최적 체크포인트")
    print("="*70)
    
    # 1. 성공률 기준
    best_by_success = df.loc[df["mean_success_rate"].idxmax()]
    print(f"\n🥇 최고 성공률: {best_by_success['checkpoint']}")
    print(f"   스텝: {best_by_success['steps']:,}")
    print(f"   성공률: {best_by_success['mean_success_rate']:.1f}%")
    print(f"   보상: {best_by_success['mean_reward']:.2f}")
    
    # 2. 보상 기준
    best_by_reward = df.loc[df["mean_reward"].idxmax()]
    print(f"\n🥈 최고 보상: {best_by_reward['checkpoint']}")
    print(f"   스텝: {best_by_reward['steps']:,}")
    print(f"   보상: {best_by_reward['mean_reward']:.2f}")
    print(f"   성공률: {best_by_reward['mean_success_rate']:.1f}%")
    
    # 3. 균형 기준 (성공률 + 보상의 정규화된 합)
    df["normalized_success"] = (df["mean_success_rate"] - df["mean_success_rate"].min()) / (df["mean_success_rate"].max() - df["mean_success_rate"].min())
    df["normalized_reward"] = (df["mean_reward"] - df["mean_reward"].min()) / (df["mean_reward"].max() - df["mean_reward"].min())
    df["balanced_score"] = df["normalized_success"] * 0.6 + df["normalized_reward"] * 0.4
    
    best_balanced = df.loc[df["balanced_score"].idxmax()]
    print(f"\n🏅 최고 균형 성능: {best_balanced['checkpoint']}")
    print(f"   스텝: {best_balanced['steps']:,}")
    print(f"   성공률: {best_balanced['mean_success_rate']:.1f}%")
    print(f"   보상: {best_balanced['mean_reward']:.2f}")
    print(f"   균형 점수: {best_balanced['balanced_score']:.3f}")
    
    # 결과 저장 (NumPy 타입을 Python 타입으로 변환)
    output_file = "results/checkpoint_analysis.json"
    os.makedirs("results", exist_ok=True)
    
    # NumPy 타입을 Python 타입으로 변환하는 함수
    def convert_to_python_types(obj):
        if isinstance(obj, dict):
            return {k: convert_to_python_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_python_types(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
    with open(output_file, "w") as f:
        json.dump({
            "all_results": convert_to_python_types(results),
            "best_by_success": convert_to_python_types(best_by_success.to_dict()),
            "best_by_reward": convert_to_python_types(best_by_reward.to_dict()),
            "best_balanced": convert_to_python_types(best_balanced.to_dict())
        }, f, indent=2)
    
    print(f"\n💾 결과 저장: {output_file}")
    
    # CSV로도 저장
    csv_file = "results/checkpoint_analysis.csv"
    df.to_csv(csv_file, index=False)
    print(f"💾 CSV 저장: {csv_file}")
    
    # 과적합 분석
    print("\n" + "="*70)
    print("📉 과적합 분석")
    print("="*70)
    
    # 성공률 추세
    if len(df) > 1:
        early_success = df.iloc[:len(df)//3]["mean_success_rate"].mean()
        mid_success = df.iloc[len(df)//3:2*len(df)//3]["mean_success_rate"].mean()
        late_success = df.iloc[2*len(df)//3:]["mean_success_rate"].mean()
        
        print(f"\n성공률 추세:")
        print(f"  초반 (0-33%): {early_success:.1f}%")
        print(f"  중반 (33-66%): {mid_success:.1f}%")
        print(f"  후반 (66-100%): {late_success:.1f}%")
        
        if late_success < mid_success:
            print(f"\n⚠️  후반부에 성능 저하 발견! ({mid_success:.1f}% → {late_success:.1f}%)")
            print(f"   과적합 가능성이 높습니다.")
            print(f"   추천: 중반 체크포인트 사용")
        else:
            print(f"\n✅ 성능이 지속적으로 향상되고 있습니다.")
            print(f"   추천: 최종 모델 또는 최고 성능 체크포인트 사용")
    
    print("\n" + "="*70)
    print("✅ 분석 완료!")
    print("="*70)


if __name__ == "__main__":
    main()

