"""
평가 결과 디버깅 스크립트

평가 결과 JSON 파일의 내용을 확인하고 문제를 진단합니다.
"""

import json
import os
import sys

def debug_results(results_file):
    """
    평가 결과 파일 디버깅
    
    Args:
        results_file: 평가 결과 JSON 파일 경로
    """
    print("\n" + "="*60)
    print("🔍 평가 결과 디버깅")
    print("="*60)
    
    # 파일 존재 확인
    if not os.path.exists(results_file):
        print(f"❌ 파일을 찾을 수 없습니다: {results_file}")
        print("\n💡 해결 방법:")
        print("1. 먼저 평가를 실행하세요:")
        print("   python evaluate.py --model models/your_model.zip")
        print("2. 평가 결과가 생성되었는지 확인하세요:")
        print("   ls -la results/")
        return
    
    # 파일 로드
    try:
        with open(results_file, 'r') as f:
            results = json.load(f)
        print(f"✅ 파일 로드 성공: {results_file}\n")
    except Exception as e:
        print(f"❌ 파일 로드 실패: {e}")
        return
    
    # 기본 정보
    print("📋 기본 정보:")
    print(f"  모델: {results.get('model_path', 'N/A')}")
    print(f"  테스트 시드: {results.get('test_seeds', 'N/A')}")
    print(f"  에피소드/시드: {results.get('n_episodes', 'N/A')}")
    
    # 시드별 결과 확인
    seed_results = results.get("seed_results", {})
    print(f"\n📊 시드별 결과 ({len(seed_results)}개):")
    
    if not seed_results:
        print("  ❌ 시드 결과가 없습니다!")
        return
    
    all_success_rates = []
    
    for seed, stats in seed_results.items():
        success_rate = stats.get("success_rate", 0) * 100
        all_success_rates.append(success_rate)
        
        print(f"\n  시드 {seed}:")
        print(f"    평균 보상: {stats.get('mean_reward', 0):.2f}")
        print(f"    성공률: {success_rate:.1f}%")
        print(f"    충돌률: {stats.get('crash_rate', 0)*100:.1f}%")
        print(f"    도로 이탈률: {stats.get('out_of_road_rate', 0)*100:.1f}%")
        
        # 에피소드 보상 확인
        episode_rewards = stats.get("episode_rewards", [])
        if episode_rewards:
            print(f"    에피소드 수: {len(episode_rewards)}")
            print(f"    보상 범위: [{min(episode_rewards):.2f}, {max(episode_rewards):.2f}]")
    
    # 성공률 문제 진단
    print("\n" + "="*60)
    print("🔍 성공률 진단:")
    print("="*60)
    
    max_success_rate = max(all_success_rates) if all_success_rates else 0
    min_success_rate = min(all_success_rates) if all_success_rates else 0
    avg_success_rate = sum(all_success_rates) / len(all_success_rates) if all_success_rates else 0
    
    print(f"  최대 성공률: {max_success_rate:.1f}%")
    print(f"  최소 성공률: {min_success_rate:.1f}%")
    print(f"  평균 성공률: {avg_success_rate:.1f}%")
    
    if max_success_rate == 0:
        print("\n❌ 문제 발견: 모든 시드에서 성공률이 0%입니다!")
        print("\n💡 가능한 원인:")
        print("1. 학습이 충분하지 않음")
        print("   → 더 오래 학습하세요 (예: 100K+ 스텝)")
        print("2. 모델이 제대로 학습되지 않음")
        print("   → TensorBoard로 학습 곡선 확인")
        print("   → tensorboard --logdir logs/")
        print("3. 평가 환경이 학습 환경과 다름")
        print("   → config.py의 환경 설정 확인")
        print("\n💡 해결 방법:")
        print("1. 학습 상태 확인:")
        print("   tensorboard --logdir logs/")
        print("2. 더 긴 학습:")
        print("   python train.py --mode fixed")
        print("3. 빠른 테스트로 환경 확인:")
        print("   python quick_start.py --demo")
    elif max_success_rate < 10:
        print("\n⚠️  경고: 성공률이 매우 낮습니다 (< 10%)")
        print("\n💡 권장 사항:")
        print("1. 학습을 더 진행하세요")
        print("2. 하이퍼파라미터 조정을 고려하세요")
        print("3. 학습 곡선을 확인하세요")
    elif max_success_rate < 50:
        print("\n⚠️  주의: 성공률이 낮습니다 (< 50%)")
        print("더 많은 학습이 필요할 수 있습니다.")
    else:
        print("\n✅ 성공률이 양호합니다!")
    
    # 시각화 가능 여부
    print("\n" + "="*60)
    print("📈 시각화 상태:")
    print("="*60)
    
    if max_success_rate > 0:
        print("✅ Success Rate 그래프가 표시됩니다")
        print("   (막대 위에 정확한 값이 표시됩니다)")
    else:
        print("⚠️  Success Rate 그래프에 막대가 보이지 않을 수 있습니다")
        print("   (하지만 수정된 코드로 0% 값이 표시됩니다)")
    
    print("\n💡 시각화 실행:")
    print(f"  python visualize.py --results {results_file}")
    
    print("\n" + "="*60 + "\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="평가 결과 디버깅")
    parser.add_argument("--results", type=str, 
                       default="results/evaluation_results.json",
                       help="평가 결과 JSON 파일")
    
    args = parser.parse_args()
    
    debug_results(args.results)
