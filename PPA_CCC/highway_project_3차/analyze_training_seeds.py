"""
학습 시드 난이도 분석 스크립트

850k 모델로 학습 시드 6개를 각각 평가하여 난이도를 측정합니다.
"""

import os
import sys
import numpy as np
from tqdm import tqdm
from stable_baselines3 import PPO
from config import TRAIN_SEEDS, FIXED_SEED_ENV_CONFIG
from envs.metadrive_env import make_env

def evaluate_seed(model, seed, n_episodes=20):
    """특정 시드에서 모델 평가"""
    # 환경 설정
    env_config = FIXED_SEED_ENV_CONFIG.copy()
    env_config["start_seed"] = seed
    env_config["num_scenarios"] = 1
    
    # 환경 생성
    env = make_env(seed=seed, render=False, config=env_config)()
    
    rewards = []
    successes = []
    crashes = []
    lengths = []
    
    for _ in tqdm(range(n_episodes), desc=f"시드 {seed}", leave=False):
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        episode_length = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            episode_length += 1
        
        rewards.append(episode_reward)
        lengths.append(episode_length)
        successes.append(info.get("arrive_dest", False))
        crashes.append(info.get("crash", False) or info.get("crash_vehicle", False))
    
    env.close()
    
    return {
        "seed": seed,
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "success_rate": float(np.mean(successes) * 100),
        "crash_rate": float(np.mean(crashes) * 100),
        "mean_length": float(np.mean(lengths)),
    }

def main():
    model_path = "models/ppo_multi_seed_6seeds_v3_750000_steps.zip"
    
    print("\n" + "="*70)
    print("🔍 학습 시드 난이도 분석")
    print("="*70)
    print(f"모델: {model_path}")
    print(f"학습 시드: {TRAIN_SEEDS}")
    print(f"에피소드/시드: 20")
    print("="*70 + "\n")
    
    # 모델 로드
    print("📦 모델 로딩 중...")
    model = PPO.load(model_path)
    print("✅ 모델 로드 완료\n")
    
    # 각 시드 평가
    results = []
    for seed in TRAIN_SEEDS:
        print(f"\n🎯 시드 {seed} 평가 중...")
        result = evaluate_seed(model, seed, n_episodes=20)
        results.append(result)
        
        print(f"  평균 보상: {result['mean_reward']:.2f} ± {result['std_reward']:.2f}")
        print(f"  성공률: {result['success_rate']:.1f}%")
        print(f"  충돌률: {result['crash_rate']:.1f}%")
        print(f"  평균 길이: {result['mean_length']:.1f}")
    
    # 결과 요약
    print("\n" + "="*70)
    print("📊 학습 시드 난이도 분석 결과")
    print("="*70)
    
    # 난이도별 분류
    easy_seeds = []
    medium_seeds = []
    hard_seeds = []
    
    for r in results:
        if r['success_rate'] >= 20:
            easy_seeds.append(r)
        elif r['success_rate'] >= 10:
            medium_seeds.append(r)
        else:
            hard_seeds.append(r)
    
    print(f"\n🟢 쉬운 시드 ({len(easy_seeds)}개): 성공률 ≥ 20%")
    for r in sorted(easy_seeds, key=lambda x: x['success_rate'], reverse=True):
        print(f"  시드 {r['seed']}: {r['success_rate']:.1f}% (보상: {r['mean_reward']:.1f})")
    
    print(f"\n🟡 중간 시드 ({len(medium_seeds)}개): 10% ≤ 성공률 < 20%")
    for r in sorted(medium_seeds, key=lambda x: x['success_rate'], reverse=True):
        print(f"  시드 {r['seed']}: {r['success_rate']:.1f}% (보상: {r['mean_reward']:.1f})")
    
    print(f"\n🔴 어려운 시드 ({len(hard_seeds)}개): 성공률 < 10%")
    for r in sorted(hard_seeds, key=lambda x: x['success_rate'], reverse=True):
        print(f"  시드 {r['seed']}: {r['success_rate']:.1f}% (보상: {r['mean_reward']:.1f})")
    
    # 전체 통계
    all_success_rates = [r['success_rate'] for r in results]
    all_rewards = [r['mean_reward'] for r in results]
    
    print("\n" + "-"*70)
    print("📈 전체 통계 (학습 시드 6개)")
    print("-"*70)
    print(f"평균 성공률: {np.mean(all_success_rates):.1f}% ± {np.std(all_success_rates):.1f}%")
    print(f"평균 보상: {np.mean(all_rewards):.1f} ± {np.std(all_rewards):.1f}")
    print(f"최고 성공률: {max(all_success_rates):.1f}% (시드 {results[np.argmax(all_success_rates)]['seed']})")
    print(f"최저 성공률: {min(all_success_rates):.1f}% (시드 {results[np.argmin(all_success_rates)]['seed']})")
    
    # 추천 사항
    print("\n" + "="*70)
    print("💡 추천 사항")
    print("="*70)
    
    if len(hard_seeds) >= 3:
        print("\n⚠️  어려운 시드가 많습니다 (3개 이상)")
        print("   → 어려운 시드를 제거하고 재학습 권장")
        print(f"   → 제거 대상: {[r['seed'] for r in hard_seeds]}")
        print(f"   → 유지 대상: {[r['seed'] for r in easy_seeds + medium_seeds]}")
    elif len(easy_seeds) >= 4:
        print("\n✅ 대부분의 시드가 잘 학습되었습니다!")
        print("   → 현재 설정 유지 권장")
        print("   → 850k 스텝이 최적입니다")
    else:
        print("\n🔄 균형잡힌 난이도 분포입니다")
        print("   → 현재 시드 조합 유지")
        print("   → 하이퍼파라미터 튜닝 고려")
    
    # 기존 시드 vs 새 시드 비교
    original_seeds = [1409, 2824, 5506]
    new_seeds = [4232, 5140, 6339]
    
    original_results = [r for r in results if r['seed'] in original_seeds]
    new_results = [r for r in results if r['seed'] in new_seeds]
    
    print("\n" + "-"*70)
    print("🔍 기존 시드 vs 새 시드 비교")
    print("-"*70)
    print(f"기존 시드 [1409, 2824, 5506]:")
    print(f"  평균 성공률: {np.mean([r['success_rate'] for r in original_results]):.1f}%")
    print(f"  평균 보상: {np.mean([r['mean_reward'] for r in original_results]):.1f}")
    
    print(f"\n새 시드 [4232, 5140, 6339]:")
    print(f"  평균 성공률: {np.mean([r['success_rate'] for r in new_results]):.1f}%")
    print(f"  평균 보상: {np.mean([r['mean_reward'] for r in new_results]):.1f}")
    
    original_avg = np.mean([r['success_rate'] for r in original_results])
    new_avg = np.mean([r['success_rate'] for r in new_results])
    
    if original_avg > new_avg * 1.5:
        print("\n⚠️  새 시드가 기존 시드보다 훨씬 어렵습니다!")
        print("   → 새 시드 제거 고려")
    elif new_avg > original_avg * 1.5:
        print("\n✅ 새 시드가 기존 시드보다 쉽습니다!")
        print("   → 좋은 다양성 확보")
    else:
        print("\n✅ 기존 시드와 새 시드의 난이도가 비슷합니다")
        print("   → 균형잡힌 학습 데이터")
    
    print("\n" + "="*70 + "\n")

if __name__ == "__main__":
    main()

