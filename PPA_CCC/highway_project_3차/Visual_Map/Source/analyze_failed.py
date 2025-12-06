"""
실패한 시드 자동 분석

evaluation_results.json을 읽어서 성공률이 낮은 시드의 궤적을 자동으로 시각화
"""

import json
import os
import sys


def analyze_failed_seeds(evaluation_results_path, model_path, threshold=0.3):
    """
    성공률이 낮은 시드를 찾아서 궤적 분석
    
    Args:
        evaluation_results_path: 평가 결과 JSON 경로
        model_path: 모델 경로
        threshold: 성공률 임계값 (이 값 이하면 분석)
    """
    # 평가 결과 로드
    with open(evaluation_results_path, 'r') as f:
        results = json.load(f)
    
    print("\n" + "="*60)
    print("🔍 실패 시드 자동 분석")
    print("="*60)
    print(f"평가 결과: {evaluation_results_path}")
    print(f"모델: {model_path}")
    print(f"임계값: 성공률 {threshold*100:.0f}% 이하")
    print("="*60 + "\n")
    
    # 성공률이 낮은 시드 찾기
    failed_seeds = []
    seed_results = results["seed_results"]
    
    print("📊 시드별 성공률:\n")
    for seed, stats in seed_results.items():
        success_rate = stats["success_rate"]
        status = "❌ 실패" if success_rate <= threshold else "✅ 양호"
        print(f"  Seed {seed:4s}: {success_rate*100:5.1f}% {status}")
        
        if success_rate <= threshold:
            failed_seeds.append(int(seed))
    
    if not failed_seeds:
        print(f"\n✅ 모든 시드가 {threshold*100:.0f}% 이상의 성공률을 보입니다!")
        return
    
    print(f"\n🎯 분석 대상 시드: {failed_seeds}")
    print(f"   총 {len(failed_seeds)}개 시드\n")
    
    # 궤적 분석 실행
    print("="*60)
    print("🚗 궤적 분석 시작")
    print("="*60 + "\n")
    
    for seed in failed_seeds:
        print(f"\n{'='*60}")
        print(f"Seed {seed} 분석 중...")
        print(f"{'='*60}")
        
        # visualize_trajectory.py를 import하여 실행
        from visualize_trajectory import evaluate_with_trajectory
        
        output_dir = f"results/failed_seeds/seed_{seed}"
        
        try:
            evaluate_with_trajectory(
                model_path=model_path,
                seed=seed,
                n_episodes=5,
                save_dir=output_dir
            )
        except Exception as e:
            print(f"❌ 오류 발생: {e}")
            continue
    
    # 요약 리포트 생성
    create_summary_report(failed_seeds, seed_results)


def create_summary_report(failed_seeds, seed_results):
    """
    실패 시드 요약 리포트 생성
    
    Args:
        failed_seeds: 실패한 시드 리스트
        seed_results: 시드별 결과
    """
    report_path = "results/failed_seeds/ANALYSIS_REPORT.txt"
    os.makedirs("results/failed_seeds", exist_ok=True)
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("실패 시드 분석 리포트\n")
        f.write("="*60 + "\n\n")
        
        for seed in failed_seeds:
            stats = seed_results[str(seed)]
            
            f.write(f"Seed {seed}\n")
            f.write("-"*60 + "\n")
            f.write(f"성공률:        {stats['success_rate']*100:.1f}%\n")
            f.write(f"충돌률:        {stats['crash_rate']*100:.1f}%\n")
            f.write(f"도로 이탈률:   {stats['out_of_road_rate']*100:.1f}%\n")
            f.write(f"평균 보상:     {stats['mean_reward']:.2f}\n")
            f.write(f"평균 길이:     {stats['mean_length']:.1f} steps\n")
            f.write(f"\n주요 실패 원인:\n")
            
            if stats['out_of_road_rate'] > 0.5:
                f.write("  ⚠️  도로 이탈이 주요 원인 (50% 이상)\n")
                f.write("     → 차선 유지 능력 개선 필요\n")
            
            if stats['crash_rate'] > 0.3:
                f.write("  ⚠️  충돌이 빈번함 (30% 이상)\n")
                f.write("     → 장애물 회피 능력 개선 필요\n")
            
            if stats['mean_length'] < 100:
                f.write("  ⚠️  에피소드가 매우 짧음 (<100 steps)\n")
                f.write("     → 초기 주행 안정성 문제\n")
            
            f.write(f"\n궤적 이미지: results/failed_seeds/seed_{seed}.png/\n")
            f.write("\n" + "="*60 + "\n\n")
    
    print(f"\n📄 리포트 저장: {report_path}")


def quick_visualize_worst_seed(evaluation_results_path, model_path):
    """
    가장 성능이 안 좋은 시드 하나만 빠르게 시각화
    
    Args:
        evaluation_results_path: 평가 결과 JSON 경로
        model_path: 모델 경로
    """
    with open(evaluation_results_path, 'r') as f:
        results = json.load(f)
    
    # 성공률이 가장 낮은 시드 찾기
    seed_results = results["seed_results"]
    worst_seed = None
    lowest_success_rate = 1.0
    
    for seed, stats in seed_results.items():
        if stats["success_rate"] < lowest_success_rate:
            lowest_success_rate = stats["success_rate"]
            worst_seed = int(seed)
    
    print("\n" + "="*60)
    print(f"🎯 최악의 성능 시드: {worst_seed}")
    print(f"   성공률: {lowest_success_rate*100:.1f}%")
    print("="*60 + "\n")
    
    from visualize_trajectory import evaluate_with_trajectory
    
    evaluate_with_trajectory(
        model_path=model_path,
        seed=worst_seed,
        n_episodes=5,
        save_dir=f"results/worst_seed_{worst_seed}"
    )


if __name__ == "__main__":
    """
    사용법:
        # 자동 분석 (성공률 30% 이하)
        python analyze_failed.py
        
        # 임계값 변경 (50% 이하)
        python analyze_failed.py --threshold 0.5
        
        # 최악의 시드만 빠르게 확인
        python analyze_failed.py --quick
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="실패 시드 자동 분석")
    parser.add_argument("--results", type=str, 
                       default="results/evaluation_results.json",
                       help="평가 결과 JSON 경로")
    parser.add_argument("--model", type=str,
                       default="models/sac_fixed_seed_1000.zip",
                       help="모델 경로")
    parser.add_argument("--threshold", type=float, default=0.3,
                       help="성공률 임계값 (기본: 0.3)")
    parser.add_argument("--quick", action="store_true",
                       help="최악의 시드만 빠르게 확인")
    
    args = parser.parse_args()
    
    # 파일 존재 확인
    if not os.path.exists(args.results):
        print(f"❌ 평가 결과를 찾을 수 없습니다: {args.results}")
        print("   먼저 'python evaluate.py'를 실행하세요")
        sys.exit(1)
    
    if not os.path.exists(args.model):
        print(f"❌ 모델을 찾을 수 없습니다: {args.model}")
        sys.exit(1)
    
    # 실행
    if args.quick:
        quick_visualize_worst_seed(args.results, args.model)
    else:
        analyze_failed_seeds(args.results, args.model, args.threshold)