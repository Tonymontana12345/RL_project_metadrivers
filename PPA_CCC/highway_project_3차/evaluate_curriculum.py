"""
커리큘럼 학습 모델 전용 평가 스크립트

커리큘럼 최종 단계(Stage 8)와 동일한 환경에서 평가합니다.
"""

import argparse
import numpy as np
from sb3_contrib import RecurrentPPO
from config import TEST_SEEDS
from config_ppo_curriculum_v2 import CURRICULUM_STAGES, BASE_ENV_CONFIG
from envs.metadrive_env import make_env
from tqdm import tqdm


def evaluate_curriculum_model(model_path, test_seeds, n_episodes=20, render=False):
    """
    커리큘럼 모델을 Stage 8 환경에서 평가
    
    Args:
        model_path: 모델 파일 경로
        test_seeds: 테스트할 시드 리스트
        n_episodes: 각 시드당 에피소드 수
        render: 렌더링 여부
    """
    print("\n" + "="*70)
    print("📊 커리큘럼 학습 모델 평가 (Stage 8 환경)")
    print("="*70)
    print(f"모델: {model_path}")
    print(f"테스트 시드: {test_seeds}")
    print(f"에피소드/시드: {n_episodes}")
    print("="*70 + "\n")
    
    # 모델 로드
    print("🔄 RecurrentPPO 모델 로드 중...")
    try:
        model = RecurrentPPO.load(model_path)
        print("✅ 모델 로드 완료\n")
    except Exception as e:
        print(f"❌ 모델 로드 실패: {e}")
        return None
    
    # Stage 8 환경 설정 사용
    stage8_config = CURRICULUM_STAGES["stage_8"]
    eval_env_config = BASE_ENV_CONFIG.copy()
    eval_env_config.update({
        "map": stage8_config["map"],  # TSCO
        "traffic_density": stage8_config["traffic_density"],  # 0.15
        "random_traffic": stage8_config["random_traffic"],  # True
        "decision_repeat": stage8_config["decision_repeat"],  # 2
        "horizon": stage8_config["horizon"],  # 2000
        "num_scenarios": 1,  # 평가는 각 시드당 1개 시나리오
    })
    
    print("🌍 평가 환경 설정:")
    print(f"   ├─ 맵: {eval_env_config['map']}")
    print(f"   ├─ 교통 밀도: {eval_env_config['traffic_density']}")
    print(f"   ├─ 랜덤 교통: {eval_env_config['random_traffic']}")
    print(f"   ├─ decision_repeat: {eval_env_config['decision_repeat']}")
    print(f"   └─ horizon: {eval_env_config['horizon']}")
    print()
    
    # 결과 저장
    all_results = {
        "model_path": model_path,
        "test_seeds": test_seeds,
        "n_episodes": n_episodes,
        "seed_results": {}
    }
    
    # 각 시드별 평가
    for seed in test_seeds:
        print(f"\n🎯 시드 {seed} 평가 중...")
        
        # 환경 생성
        env = make_env(seed=seed, render=render, config=eval_env_config)()
        
        # 에피소드별 결과
        episode_rewards = []
        episode_lengths = []
        success_count = 0
        crash_count = 0
        out_of_road_count = 0
        max_step_count = 0
        
        # 진행 바
        for episode in tqdm(range(n_episodes), desc=f"Seed {seed}"):
            obs, info = env.reset()
            done = False
            total_reward = 0
            steps = 0
            
            # RecurrentPPO를 위한 LSTM state 초기화
            lstm_states = None
            episode_start = np.ones((1,), dtype=bool)
            
            while not done:
                action, lstm_states = model.predict(
                    obs, 
                    state=lstm_states,
                    episode_start=episode_start,
                    deterministic=True
                )
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                total_reward += reward
                steps += 1
                episode_start = np.zeros((1,), dtype=bool)  # 에피소드 시작은 첫 스텝만
            
            # 결과 기록
            episode_rewards.append(total_reward)
            episode_lengths.append(steps)
            
            # 종료 이유 분석
            if info.get("arrive_dest", False):
                success_count += 1
            if info.get("crash", False) or info.get("crash_vehicle", False):
                crash_count += 1
            if info.get("out_of_road", False):
                out_of_road_count += 1
            if steps >= eval_env_config["horizon"] - 1:
                max_step_count += 1
        
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
            "max_step_rate": max_step_count / n_episodes,
        }
        
        all_results["seed_results"][seed] = seed_stats
        
        # 시드별 결과 출력
        print(f"\n  평균 보상: {seed_stats['mean_reward']:.2f} ± {seed_stats['std_reward']:.2f}")
        print(f"  평균 길이: {seed_stats['mean_length']:.1f} 스텝")
        print(f"  성공률: {seed_stats['success_rate']*100:.1f}%")
        print(f"  충돌률: {seed_stats['crash_rate']*100:.1f}%")
        print(f"  도로이탈률: {seed_stats['out_of_road_rate']*100:.1f}%")
        print(f"  최대스텝률: {seed_stats['max_step_rate']*100:.1f}%")
    
    # 전체 요약
    print("\n" + "="*70)
    print("📈 전체 요약")
    print("="*70)
    
    all_rewards = [s["mean_reward"] for s in all_results["seed_results"].values()]
    all_success_rates = [s["success_rate"] for s in all_results["seed_results"].values()]
    all_lengths = [s["mean_length"] for s in all_results["seed_results"].values()]
    
    print(f"전체 평균 보상: {np.mean(all_rewards):.2f} ± {np.std(all_rewards):.2f}")
    print(f"전체 평균 길이: {np.mean(all_lengths):.1f} ± {np.std(all_lengths):.1f} 스텝")
    print(f"전체 평균 성공률: {np.mean(all_success_rates)*100:.1f}% ± {np.std(all_success_rates)*100:.1f}%")
    
    # 성공률 분석
    if np.mean(all_success_rates) == 0:
        print("\n⚠️  경고: 성공률이 0%입니다!")
        print("💡 가능한 원인:")
        print("   1. 학습이 충분하지 않음")
        print("   2. Stage 8 환경이 너무 어려움")
        print("   3. 더 긴 학습 또는 하이퍼파라미터 조정 필요")
    elif np.mean(all_success_rates) < 0.3:
        print("\n⚠️  성공률이 낮습니다 (30% 미만)")
        print("💡 개선 방법:")
        print("   1. 더 많은 스텝으로 학습")
        print("   2. 보상 체계 재조정")
        print("   3. Stage별 학습 스텝 증가")
    else:
        print("\n✅ 학습이 성공적으로 완료되었습니다!")
    
    print("="*70 + "\n")
    
    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="커리큘럼 학습 모델 평가")
    parser.add_argument("--model", type=str, 
                       default="models/ppo_8_stage_curriculum_v1_final.zip",
                       help="평가할 모델 경로")
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                       help="테스트 시드 (기본: config의 TEST_SEEDS)")
    parser.add_argument("--episodes", type=int, default=20,
                       help="각 시드당 에피소드 수")
    parser.add_argument("--render", action="store_true",
                       help="렌더링 활성화")
    
    args = parser.parse_args()
    
    # 테스트 시드 설정
    test_seeds = args.seeds if args.seeds else TEST_SEEDS
    
    # 평가 실행
    results = evaluate_curriculum_model(
        model_path=args.model,
        test_seeds=test_seeds,
        n_episodes=args.episodes,
        render=args.render
    )
