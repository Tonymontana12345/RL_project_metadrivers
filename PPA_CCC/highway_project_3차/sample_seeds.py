"""
랜덤 시드 샘플링 및 난이도 평가 스크립트

750k 최고 모델로 랜덤 시드 30개를 평가하여 적절한 난이도의 시드를 찾습니다.
"""

import os
import sys
import numpy as np
import random
from tqdm import tqdm
from stable_baselines3 import PPO
from config import TEST_SEEDS, FIXED_SEED_ENV_CONFIG
from envs.metadrive_env import make_env

# 테스트 시드와 현재 학습 시드 (중복 방지)
EXISTING_SEEDS = [1409, 2824, 5506, 6339] + TEST_SEEDS

def evaluate_seed(model, seed, n_episodes=5):
    """특정 시드에서 모델 평가 (빠른 평가를 위해 5 에피소드만)"""
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
    
    for _ in range(n_episodes):
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

def generate_random_seeds(n_seeds=30, min_seed=1000, max_seed=10000):
    """중복되지 않는 랜덤 시드 생성"""
    seeds = []
    while len(seeds) < n_seeds:
        seed = random.randint(min_seed, max_seed)
        if seed not in EXISTING_SEEDS and seed not in seeds:
            seeds.append(seed)
    return seeds

def main():
    model_path = "models/ppo_multi_seed_6seeds_v3_750000_steps.zip"
    n_candidate_seeds = 30
    
    print("\n" + "="*70)
    print("🔍 랜덤 시드 샘플링 및 난이도 평가")
    print("="*70)
    print(f"모델: {model_path}")
    print(f"평가할 시드 개수: {n_candidate_seeds}")
    print(f"에피소드/시드: 5 (빠른 평가)")
    print(f"제외할 시드: {EXISTING_SEEDS}")
    print("="*70 + "\n")
    
    # 모델 로드
    print("📦 모델 로딩 중...")
    if not os.path.exists(model_path):
        print(f"❌ 모델을 찾을 수 없습니다: {model_path}")
        print("   750k 모델이 없습니다. 800k 모델을 사용하시겠습니까? (y/n)")
        return
    
    model = PPO.load(model_path)
    print("✅ 모델 로드 완료\n")
    
    # 랜덤 시드 생성
    print(f"🎲 랜덤 시드 {n_candidate_seeds}개 생성 중...")
    candidate_seeds = generate_random_seeds(n_candidate_seeds)
    print(f"✅ 생성 완료: {candidate_seeds[:10]}... (처음 10개)\n")
    
    # 각 시드 평가
    print("🎯 시드 평가 시작...\n")
    results = []
    for i, seed in enumerate(tqdm(candidate_seeds, desc="시드 평가")):
        result = evaluate_seed(model, seed, n_episodes=5)
        results.append(result)
        
        # 진행 상황 출력 (10개마다)
        if (i + 1) % 10 == 0:
            print(f"\n진행: {i+1}/{n_candidate_seeds} 완료")
    
    print("\n✅ 모든 시드 평가 완료!\n")
    
    # 결과 분류
    print("="*70)
    print("📊 시드 난이도 분류 결과")
    print("="*70)
    
    easy_seeds = []
    medium_seeds = []
    hard_seeds = []
    
    for r in results:
        if r['success_rate'] >= 30:
            easy_seeds.append(r)
        elif r['success_rate'] >= 15:
            medium_seeds.append(r)
        else:
            hard_seeds.append(r)
    
    # 쉬운 시드
    print(f"\n🟢 쉬운 시드 ({len(easy_seeds)}개): 성공률 ≥ 30%")
    print("-"*70)
    if easy_seeds:
        for r in sorted(easy_seeds, key=lambda x: x['success_rate'], reverse=True):
            print(f"  시드 {r['seed']:5d}: 성공률 {r['success_rate']:5.1f}% | "
                  f"보상 {r['mean_reward']:6.1f} | 충돌률 {r['crash_rate']:5.1f}%")
    else:
        print("  없음")
    
    # 중간 시드
    print(f"\n🟡 중간 시드 ({len(medium_seeds)}개): 15% ≤ 성공률 < 30%")
    print("-"*70)
    if medium_seeds:
        for r in sorted(medium_seeds, key=lambda x: x['success_rate'], reverse=True):
            print(f"  시드 {r['seed']:5d}: 성공률 {r['success_rate']:5.1f}% | "
                  f"보상 {r['mean_reward']:6.1f} | 충돌률 {r['crash_rate']:5.1f}%")
    else:
        print("  없음")
    
    # 어려운 시드
    print(f"\n🔴 어려운 시드 ({len(hard_seeds)}개): 성공률 < 15%")
    print("-"*70)
    if hard_seeds:
        for r in sorted(hard_seeds, key=lambda x: x['success_rate'], reverse=True)[:5]:
            print(f"  시드 {r['seed']:5d}: 성공률 {r['success_rate']:5.1f}% | "
                  f"보상 {r['mean_reward']:6.1f} | 충돌률 {r['crash_rate']:5.1f}%")
        if len(hard_seeds) > 5:
            print(f"  ... 외 {len(hard_seeds)-5}개 더")
    else:
        print("  없음")
    
    # 추천 시드
    print("\n" + "="*70)
    print("💡 추천 시드")
    print("="*70)
    
    print("\n현재 유지할 시드 (4개):")
    print("  [1409, 2824, 5506, 6339]")
    print("  - 쉬움: 6339")
    print("  - 중간: 1409")
    print("  - 어려움: 2824, 5506")
    
    print("\n추가 필요:")
    print("  - 쉬운 시드 1개 (30% 이상)")
    print("  - 중간 시드 1개 (15-30%)")
    
    if easy_seeds and medium_seeds:
        best_easy = sorted(easy_seeds, key=lambda x: x['success_rate'], reverse=True)[0]
        best_medium = sorted(medium_seeds, key=lambda x: x['success_rate'], reverse=True)[0]
        
        print("\n✅ 추천 추가 시드:")
        print(f"  🟢 쉬운 시드: {best_easy['seed']} (성공률 {best_easy['success_rate']:.1f}%)")
        print(f"  🟡 중간 시드: {best_medium['seed']} (성공률 {best_medium['success_rate']:.1f}%)")
        
        print("\n최종 학습 시드 (6개):")
        final_seeds = [1409, 2824, 5506, 6339, best_easy['seed'], best_medium['seed']]
        print(f"  {final_seeds}")
        print("\n난이도 분포:")
        print(f"  🟢 쉬움 (2개): [6339, {best_easy['seed']}]")
        print(f"  🟡 중간 (2개): [1409, {best_medium['seed']}]")
        print(f"  🔴 어려움 (2개): [2824, 5506]")
        
        print("\n📝 config.py에 다음과 같이 업데이트하세요:")
        print(f"  TRAIN_SEEDS = {final_seeds}")
        
    elif easy_seeds and not medium_seeds:
        print("\n⚠️  쉬운 시드는 있지만 중간 시드가 없습니다!")
        print("   옵션:")
        print("   1. 쉬운 시드 2개 추가 (5개 시드)")
        print("   2. 더 많은 시드 샘플링 (이 스크립트 다시 실행)")
        
        if len(easy_seeds) >= 2:
            top_easy = sorted(easy_seeds, key=lambda x: x['success_rate'], reverse=True)[:2]
            print(f"\n   옵션 1 추천 시드:")
            for r in top_easy:
                print(f"     🟢 {r['seed']} (성공률 {r['success_rate']:.1f}%)")
    
    elif medium_seeds and not easy_seeds:
        print("\n⚠️  중간 시드는 있지만 쉬운 시드가 없습니다!")
        print("   옵션:")
        print("   1. 중간 시드 2개 추가 (5개 시드)")
        print("   2. 더 많은 시드 샘플링 (이 스크립트 다시 실행)")
        
        if len(medium_seeds) >= 2:
            top_medium = sorted(medium_seeds, key=lambda x: x['success_rate'], reverse=True)[:2]
            print(f"\n   옵션 1 추천 시드:")
            for r in top_medium:
                print(f"     🟡 {r['seed']} (성공률 {r['success_rate']:.1f}%)")
    
    else:
        print("\n❌ 적절한 난이도의 시드를 찾지 못했습니다!")
        print("   옵션:")
        print("   1. 더 많은 시드 샘플링 (n_candidate_seeds 증가)")
        print("   2. 현재 4개 시드만 사용")
        print("   3. 난이도 기준 완화 (쉬움: 25%+, 중간: 10-25%)")
    
    print("\n" + "="*70 + "\n")

if __name__ == "__main__":
    main()

