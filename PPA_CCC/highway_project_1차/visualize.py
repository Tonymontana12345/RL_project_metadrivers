"""
결과 시각화 스크립트

학습 및 평가 결과를 그래프로 시각화합니다.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import json
import os
from glob import glob

from config import RESULTS_DIR, LOGS_DIR
from utils.path_utils import get_result_path


# 스타일 설정
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def plot_training_curve(log_dir, save_path=None):
    """
    학습 곡선 그리기 (TensorBoard 로그 사용)
    
    Args:
        log_dir: TensorBoard 로그 디렉토리
        save_path: 저장 경로
    """
    try:
        from tensorboard.backend.event_processing import event_accumulator
        
        # 이벤트 파일 찾기
        event_files = glob(os.path.join(log_dir, "**", "events.out.tfevents.*"), recursive=True)
        
        if not event_files:
            print(f"⚠️  로그 파일을 찾을 수 없습니다: {log_dir}")
            return
        
        # 데이터 로드
        ea = event_accumulator.EventAccumulator(os.path.dirname(event_files[0]))
        ea.Reload()
        
        # 보상 데이터 추출
        if 'rollout/ep_rew_mean' in ea.Tags()['scalars']:
            rewards = ea.Scalars('rollout/ep_rew_mean')
            steps = [r.step for r in rewards]
            values = [r.value for r in rewards]
            
            # 그래프 그리기
            plt.figure(figsize=(12, 6))
            plt.plot(steps, values, linewidth=2)
            plt.xlabel('Timesteps', fontsize=12)
            plt.ylabel('Mean Episode Reward', fontsize=12)
            plt.title('Training Curve', fontsize=14, fontweight='bold')
            plt.grid(True, alpha=0.3)
            
            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                print(f"💾 학습 곡선 저장: {save_path}")
            else:
                plt.show()
            
            plt.close()
        else:
            print("⚠️  보상 데이터를 찾을 수 없습니다")
            
    except ImportError:
        print("⚠️  tensorboard 패키지가 필요합니다: pip install tensorboard")
    except Exception as e:
        print(f"❌ 학습 곡선 그리기 실패: {e}")


def plot_evaluation_results(results_file, save_path=None):
    """
    평가 결과 시각화
    
    Args:
        results_file: 평가 결과 JSON 파일
        save_path: 저장 경로
    """
    # 결과 로드
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    seed_results = results["seed_results"]
    
    # 데이터 준비
    seeds = []
    mean_rewards = []
    std_rewards = []
    success_rates = []
    
    for seed, stats in seed_results.items():
        seeds.append(int(seed))
        mean_rewards.append(stats["mean_reward"])
        std_rewards.append(stats["std_reward"])
        success_rates.append(stats["success_rate"] * 100)
    
    # 그래프 생성
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # 1. 보상 비교
    ax1 = axes[0]
    x_pos = np.arange(len(seeds))
    ax1.bar(x_pos, mean_rewards, yerr=std_rewards, capsize=5, alpha=0.7)
    ax1.set_xlabel('Seed', fontsize=12)
    ax1.set_ylabel('Mean Reward', fontsize=12)
    ax1.set_title('Reward Comparison Across Seeds', fontsize=14, fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(seeds)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 2. 성공률 비교
    ax2 = axes[1]
    ax2.bar(x_pos, success_rates, alpha=0.7, color='green')
    ax2.set_xlabel('Seed', fontsize=12)
    ax2.set_ylabel('Success Rate (%)', fontsize=12)
    ax2.set_title('Success Rate Across Seeds', fontsize=14, fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(seeds)
    ax2.set_ylim([0, 100])
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"💾 평가 결과 저장: {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_episode_rewards_distribution(results_file, save_path=None):
    """
    에피소드 보상 분포 그리기
    
    Args:
        results_file: 평가 결과 JSON 파일
        save_path: 저장 경로
    """
    # 결과 로드
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    seed_results = results["seed_results"]
    
    # 데이터 준비
    data = []
    for seed, stats in seed_results.items():
        for reward in stats["episode_rewards"]:
            data.append({"Seed": int(seed), "Reward": reward})
    
    df = pd.DataFrame(data)
    
    # 박스플롯
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x="Seed", y="Reward")
    plt.xlabel('Seed', fontsize=12)
    plt.ylabel('Episode Reward', fontsize=12)
    plt.title('Episode Reward Distribution', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"💾 보상 분포 저장: {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_generalization_comparison(fixed_results, multi_results, save_path=None):
    """
    고정 시드 vs 다중 시드 일반화 성능 비교
    
    Args:
        fixed_results: 고정 시드 학습 결과 파일
        multi_results: 다중 시드 학습 결과 파일
        save_path: 저장 경로
    """
    # 결과 로드
    with open(fixed_results, 'r') as f:
        fixed = json.load(f)
    
    with open(multi_results, 'r') as f:
        multi = json.load(f)
    
    # 데이터 준비
    seeds = sorted([int(s) for s in fixed["seed_results"].keys()])
    
    fixed_rewards = [fixed["seed_results"][str(s)]["mean_reward"] for s in seeds]
    multi_rewards = [multi["seed_results"][str(s)]["mean_reward"] for s in seeds]
    
    # 그래프
    plt.figure(figsize=(12, 6))
    x_pos = np.arange(len(seeds))
    width = 0.35
    
    plt.bar(x_pos - width/2, fixed_rewards, width, label='Fixed Seed Training', alpha=0.8)
    plt.bar(x_pos + width/2, multi_rewards, width, label='Multi-Seed Training', alpha=0.8)
    
    plt.xlabel('Test Seed', fontsize=12)
    plt.ylabel('Mean Reward', fontsize=12)
    plt.title('Generalization: Fixed vs Multi-Seed Training', fontsize=14, fontweight='bold')
    plt.xticks(x_pos, seeds)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3, axis='y')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"💾 일반화 비교 저장: {save_path}")
    else:
        plt.show()
    
    plt.close()


def create_summary_report(results_file, output_file="summary_report.txt"):
    """
    요약 리포트 생성
    
    Args:
        results_file: 평가 결과 JSON 파일
        output_file: 출력 파일명
    """
    # 결과 로드
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    # 리포트 작성
    report = []
    report.append("="*60)
    report.append("PGDrive 평가 요약 리포트")
    report.append("="*60)
    report.append(f"\n모델: {results['model_path']}")
    report.append(f"테스트 시드: {results['test_seeds']}")
    report.append(f"에피소드/시드: {results['n_episodes']}")
    
    report.append("\n" + "="*60)
    report.append("전체 요약")
    report.append("="*60)
    summary = results["summary"]
    report.append(f"평균 보상: {summary['overall_mean_reward']:.2f} ± {summary['overall_std_reward']:.2f}")
    report.append(f"평균 성공률: {summary['overall_success_rate']*100:.1f}% ± {summary['success_rate_std']*100:.1f}%")
    
    report.append("\n" + "="*60)
    report.append("시드별 상세 결과")
    report.append("="*60)
    
    for seed, stats in results["seed_results"].items():
        report.append(f"\n시드 {seed}:")
        report.append(f"  평균 보상: {stats['mean_reward']:.2f} ± {stats['std_reward']:.2f}")
        report.append(f"  보상 범위: [{stats['min_reward']:.2f}, {stats['max_reward']:.2f}]")
        report.append(f"  평균 길이: {stats['mean_length']:.1f} 스텝")
        report.append(f"  성공률: {stats['success_rate']*100:.1f}%")
        report.append(f"  충돌률: {stats['crash_rate']*100:.1f}%")
        report.append(f"  도로 이탈률: {stats['out_of_road_rate']*100:.1f}%")
    
    report.append("\n" + "="*60)
    
    # 파일 저장
    output_path = get_result_path(output_file)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))
    
    print(f"💾 요약 리포트 저장: {output_path}")
    
    # 화면 출력
    print('\n'.join(report))


def visualize_all(results_file):
    """
    모든 시각화 생성
    
    Args:
        results_file: 평가 결과 JSON 파일
    """
    print("\n" + "="*60)
    print("📊 결과 시각화 시작")
    print("="*60 + "\n")
    
    base_name = os.path.splitext(os.path.basename(results_file))[0]
    
    # 1. 평가 결과
    plot_evaluation_results(
        results_file,
        save_path=get_result_path(f"{base_name}_comparison.png")
    )
    
    # 2. 보상 분포
    plot_episode_rewards_distribution(
        results_file,
        save_path=get_result_path(f"{base_name}_distribution.png")
    )
    
    # 3. 요약 리포트
    create_summary_report(
        results_file,
        output_file=f"{base_name}_report.txt"
    )
    
    print("\n" + "="*60)
    print("✅ 시각화 완료!")
    print("="*60 + "\n")


if __name__ == "__main__":
    """
    메인 실행
    
    사용법:
        # 기본 시각화
        python visualize.py
        
        # 특정 결과 파일 시각화
        python visualize.py --results results/evaluation_results.json
        
        # 학습 곡선 그리기
        python visualize.py --training-curve logs/ppo_fixed_seed_1000
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="PGDrive 결과 시각화")
    parser.add_argument("--results", type=str, default="results/evaluation_results.json",
                       help="평가 결과 JSON 파일")
    parser.add_argument("--training-curve", type=str, default=None,
                       help="학습 곡선을 그릴 로그 디렉토리")
    parser.add_argument("--compare", nargs=2, default=None,
                       help="두 결과 파일 비교 (고정 시드, 다중 시드)")
    
    args = parser.parse_args()
    
    # 학습 곡선
    if args.training_curve:
        plot_training_curve(
            args.training_curve,
            save_path=get_result_path("training_curve.png")
        )
    
    # 평가 결과 시각화
    if os.path.exists(args.results):
        visualize_all(args.results)
    else:
        print(f"⚠️  결과 파일을 찾을 수 없습니다: {args.results}")
    
    # 비교
    if args.compare:
        fixed_file, multi_file = args.compare
        if os.path.exists(fixed_file) and os.path.exists(multi_file):
            plot_generalization_comparison(
                fixed_file,
                multi_file,
                save_path=get_result_path("generalization_comparison.png")
            )
        else:
            print("⚠️  비교 파일을 찾을 수 없습니다")
