"""
앙상블 평가 결과 시각화 스크립트

앙상블 모델의 평가 결과를 다양한 차트로 시각화합니다.
"""

import json
import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, List
import seaborn as sns


def get_result_path(filename: str) -> str:
    """결과 파일 경로 생성"""
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    return str(results_dir / filename)


def load_ensemble_results(results_file: str) -> Dict:
    """앙상블 평가 결과 로드"""
    if not os.path.exists(results_file):
        # results/ 디렉토리에서 찾기
        results_file = get_result_path(os.path.basename(results_file))

    if not os.path.exists(results_file):
        raise FileNotFoundError(f"결과 파일을 찾을 수 없습니다: {results_file}")

    with open(results_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def plot_map_comparison(results: Dict, save_path: str = None):
    """
    맵별 성능 비교 차트

    성공률, 평균 보상을 맵별로 비교
    """
    map_results = results["map_results"]

    maps = []
    success_rates = []
    mean_rewards = []
    std_rewards = []

    for map_str, map_data in map_results.items():
        summary = map_data["summary"]
        maps.append(map_str)
        success_rates.append(summary["overall_success_rate"] * 100)
        mean_rewards.append(summary["overall_mean_reward"])
        std_rewards.append(summary["overall_std_reward"])

    # 2x1 서브플롯
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # 성공률 차트
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(maps)))
    bars1 = ax1.bar(maps, success_rates, color=colors, alpha=0.8, edgecolor='black')
    ax1.set_ylabel('Success Rate (%)', fontsize=12)
    ax1.set_title('Success Rate by Map', fontsize=14, fontweight='bold')
    ax1.set_ylim(0, 100)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')

    # 값 표시
    for bar, val in zip(bars1, success_rates):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}%',
                ha='center', va='bottom', fontsize=10)

    # 평균 보상 차트 (에러바 포함)
    bars2 = ax2.bar(maps, mean_rewards, yerr=std_rewards,
                    color=colors, alpha=0.8, edgecolor='black',
                    capsize=5, error_kw={'linewidth': 2})
    ax2.set_ylabel('Mean Reward', fontsize=12)
    ax2.set_title('Mean Reward by Map', fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3, linestyle='--')

    # 값 표시
    for bar, mean, std in zip(bars2, mean_rewards, std_rewards):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{mean:.1f}\n±{std:.1f}',
                ha='center', va='bottom', fontsize=9)

    plt.suptitle(f'Ensemble Performance Comparison\nStrategy: {results["strategy"]}',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 맵별 비교 차트 저장: {save_path}")
    else:
        plt.show()

    plt.close()


def plot_seed_comparison(results: Dict, save_path: str = None):
    """
    시드별 성능 비교 차트

    각 맵에 대해 시드별 성공률과 보상을 비교
    """
    map_results = results["map_results"]
    n_maps = len(map_results)

    fig, axes = plt.subplots(n_maps, 2, figsize=(14, 5 * n_maps))
    if n_maps == 1:
        axes = axes.reshape(1, -1)

    for idx, (map_str, map_data) in enumerate(map_results.items()):
        seed_results = map_data["seed_results"]

        seeds = []
        success_rates = []
        mean_rewards = []
        std_rewards = []

        for seed, seed_data in seed_results.items():
            seeds.append(seed)
            success_rates.append(seed_data["success_rate"] * 100)
            mean_rewards.append(seed_data["mean_reward"])
            std_rewards.append(seed_data["std_reward"])

        # 성공률
        ax1 = axes[idx, 0]
        colors = plt.cm.Set3(np.linspace(0, 1, len(seeds)))
        bars1 = ax1.bar(seeds, success_rates, color=colors, alpha=0.8, edgecolor='black')
        ax1.set_ylabel('Success Rate (%)', fontsize=11)
        ax1.set_xlabel('Seed', fontsize=11)
        ax1.set_title(f'{map_str} - Success Rate by Seed', fontsize=12, fontweight='bold')
        ax1.set_ylim(0, 100)
        ax1.grid(axis='y', alpha=0.3, linestyle='--')

        for bar, val in zip(bars1, success_rates):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.0f}%',
                    ha='center', va='bottom', fontsize=9)

        # 평균 보상
        ax2 = axes[idx, 1]
        bars2 = ax2.bar(seeds, mean_rewards, yerr=std_rewards,
                       color=colors, alpha=0.8, edgecolor='black',
                       capsize=5, error_kw={'linewidth': 2})
        ax2.set_ylabel('Mean Reward', fontsize=11)
        ax2.set_xlabel('Seed', fontsize=11)
        ax2.set_title(f'{map_str} - Mean Reward by Seed', fontsize=12, fontweight='bold')
        ax2.grid(axis='y', alpha=0.3, linestyle='--')

        for bar, mean, std in zip(bars2, mean_rewards, std_rewards):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{mean:.0f}\n±{std:.0f}',
                    ha='center', va='bottom', fontsize=8)

    plt.suptitle(f'Seed Comparison - {results["strategy"]} Strategy',
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 시드별 비교 차트 저장: {save_path}")
    else:
        plt.show()

    plt.close()


def plot_failure_analysis(results: Dict, save_path: str = None):
    """
    실패 원인 분석 차트

    맵별로 충돌, 도로 이탈 등 실패 원인을 분석
    """
    map_results = results["map_results"]

    maps = []
    success_rates = []
    crash_rates = []
    out_of_road_rates = []

    for map_str, map_data in map_results.items():
        summary = map_data["summary"]
        seed_results = map_data["seed_results"]

        maps.append(map_str)
        success_rates.append(summary["overall_success_rate"] * 100)

        # seed_results에서 평균 계산
        crash_avg = np.mean([sr["crash_rate"] for sr in seed_results.values()]) * 100
        out_of_road_avg = np.mean([sr["out_of_road_rate"] for sr in seed_results.values()]) * 100

        crash_rates.append(crash_avg)
        out_of_road_rates.append(out_of_road_avg)

    # 스택 바 차트
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(maps))
    width = 0.6

    p1 = ax.bar(x, success_rates, width, label='Success',
                color='#2ecc71', alpha=0.8, edgecolor='black')
    p2 = ax.bar(x, crash_rates, width, bottom=success_rates,
                label='Crash', color='#e74c3c', alpha=0.8, edgecolor='black')

    bottom = np.array(success_rates) + np.array(crash_rates)
    p3 = ax.bar(x, out_of_road_rates, width, bottom=bottom,
                label='Out of Road', color='#f39c12', alpha=0.8, edgecolor='black')

    ax.set_ylabel('Percentage (%)', fontsize=12)
    ax.set_title('Episode Outcome Distribution by Map', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(maps)
    ax.set_ylim(0, 100)
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # 값 표시
    for i, (s, c, o) in enumerate(zip(success_rates, crash_rates, out_of_road_rates)):
        if s > 5:
            ax.text(i, s/2, f'{s:.1f}%', ha='center', va='center',
                   fontsize=10, fontweight='bold', color='white')
        if c > 5:
            ax.text(i, s + c/2, f'{c:.1f}%', ha='center', va='center',
                   fontsize=10, fontweight='bold', color='white')
        if o > 5:
            ax.text(i, s + c + o/2, f'{o:.1f}%', ha='center', va='center',
                   fontsize=10, fontweight='bold', color='white')

    plt.suptitle(f'Failure Analysis - {results["strategy"]} Strategy',
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 실패 분석 차트 저장: {save_path}")
    else:
        plt.show()

    plt.close()


def plot_heatmap(results: Dict, save_path: str = None):
    """
    맵-시드 성공률 히트맵

    맵과 시드의 조합에 대한 성공률을 히트맵으로 표시
    """
    map_results = results["map_results"]

    # 데이터 준비
    maps = list(map_results.keys())
    all_seeds = set()
    for map_data in map_results.values():
        all_seeds.update(map_data["seed_results"].keys())
    seeds = sorted(all_seeds)

    # 성공률 매트릭스 생성
    success_matrix = np.zeros((len(maps), len(seeds)))

    for i, map_str in enumerate(maps):
        seed_results = map_results[map_str]["seed_results"]
        for j, seed in enumerate(seeds):
            if seed in seed_results:
                success_matrix[i, j] = seed_results[seed]["success_rate"] * 100
            else:
                success_matrix[i, j] = np.nan

    # 히트맵 생성
    fig, ax = plt.subplots(figsize=(12, len(maps) * 1.5))

    im = ax.imshow(success_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)

    # 축 설정
    ax.set_xticks(np.arange(len(seeds)))
    ax.set_yticks(np.arange(len(maps)))
    ax.set_xticklabels(seeds)
    ax.set_yticklabels(maps)

    # 축 라벨
    ax.set_xlabel('Seed', fontsize=12)
    ax.set_ylabel('Map', fontsize=12)
    ax.set_title('Success Rate Heatmap (Map × Seed)', fontsize=14, fontweight='bold')

    # 값 표시
    for i in range(len(maps)):
        for j in range(len(seeds)):
            if not np.isnan(success_matrix[i, j]):
                text = ax.text(j, i, f'{success_matrix[i, j]:.0f}%',
                             ha="center", va="center", color="black", fontsize=10)

    # 컬러바
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Success Rate (%)', rotation=270, labelpad=20, fontsize=11)

    plt.suptitle(f'Performance Heatmap - {results["strategy"]} Strategy',
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 히트맵 저장: {save_path}")
    else:
        plt.show()

    plt.close()


def plot_summary_dashboard(results: Dict, save_path: str = None):
    """
    종합 대시보드

    전체 성능을 한눈에 볼 수 있는 대시보드
    """
    summary = results["summary"]
    map_results = results["map_results"]

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # 1. 전체 요약 (텍스트)
    ax1 = fig.add_subplot(gs[0, :])
    ax1.axis('off')

    # summary 키 이름 호환성 처리
    success_rate = summary.get('overall_mean_success_rate', summary.get('overall_success_rate', 0))
    success_std = summary.get('overall_std_success_rate', summary.get('success_rate_std', 0))

    info_text = f"""
    ENSEMBLE EVALUATION SUMMARY

    Strategy: {results['strategy']}
    Models: {', '.join(results['model_names'])}

    Overall Success Rate: {success_rate*100:.1f}% ± {success_std*100:.1f}%
    Overall Mean Reward: {summary['overall_mean_reward']:.1f} ± {summary['overall_std_reward']:.1f}

    Test Maps: {', '.join(results['test_maps'])}
    Test Seeds: {', '.join(map(str, results['test_seeds']))}
    """

    ax1.text(0.5, 0.5, info_text, ha='center', va='center',
            fontsize=12, family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    # 2. 맵별 성공률
    ax2 = fig.add_subplot(gs[1, 0])
    maps = list(map_results.keys())
    success_rates = [map_results[m]["summary"]["overall_success_rate"] * 100 for m in maps]
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(maps)))
    ax2.barh(maps, success_rates, color=colors, alpha=0.8, edgecolor='black')
    ax2.set_xlabel('Success Rate (%)')
    ax2.set_title('Success Rate by Map', fontweight='bold')
    ax2.set_xlim(0, 100)
    ax2.grid(axis='x', alpha=0.3, linestyle='--')

    # 3. 맵별 평균 보상
    ax3 = fig.add_subplot(gs[1, 1])
    mean_rewards = [map_results[m]["summary"]["overall_mean_reward"] for m in maps]
    ax3.barh(maps, mean_rewards, color=colors, alpha=0.8, edgecolor='black')
    ax3.set_xlabel('Mean Reward')
    ax3.set_title('Mean Reward by Map', fontweight='bold')
    ax3.grid(axis='x', alpha=0.3, linestyle='--')

    # 4. 실패 원인 파이 차트
    ax4 = fig.add_subplot(gs[1, 2])

    # 전체 맵에서 평균 계산
    all_crash_rates = []
    all_out_of_road_rates = []
    for map_data in map_results.values():
        for seed_data in map_data["seed_results"].values():
            all_crash_rates.append(seed_data["crash_rate"])
            all_out_of_road_rates.append(seed_data["out_of_road_rate"])

    overall_success = summary.get('overall_mean_success_rate', summary.get('overall_success_rate', 0)) * 100
    overall_crash = np.mean(all_crash_rates) * 100
    overall_out_of_road = np.mean(all_out_of_road_rates) * 100

    sizes = [overall_success, overall_crash, overall_out_of_road]
    labels = ['Success', 'Crash', 'Out of Road']
    colors_pie = ['#2ecc71', '#e74c3c', '#f39c12']
    explode = (0.05, 0, 0)

    ax4.pie(sizes, explode=explode, labels=labels, colors=colors_pie,
           autopct='%1.1f%%', shadow=True, startangle=90)
    ax4.set_title('Overall Outcome Distribution', fontweight='bold')

    # 5. 맵별 상세 성능 (하단)
    ax5 = fig.add_subplot(gs[2, :])

    # 맵별 성공률, 충돌률, 도로이탈률
    x = np.arange(len(maps))
    width = 0.25

    success_data = [map_results[m]["summary"]["overall_success_rate"] * 100 for m in maps]

    # 맵별 crash_rate와 out_of_road_rate 계산
    crash_data = []
    out_data = []
    for m in maps:
        seed_results = map_results[m]["seed_results"]
        crash_avg = np.mean([sr["crash_rate"] for sr in seed_results.values()]) * 100
        out_avg = np.mean([sr["out_of_road_rate"] for sr in seed_results.values()]) * 100
        crash_data.append(crash_avg)
        out_data.append(out_avg)

    ax5.bar(x - width, success_data, width, label='Success',
           color='#2ecc71', alpha=0.8, edgecolor='black')
    ax5.bar(x, crash_data, width, label='Crash',
           color='#e74c3c', alpha=0.8, edgecolor='black')
    ax5.bar(x + width, out_data, width, label='Out of Road',
           color='#f39c12', alpha=0.8, edgecolor='black')

    ax5.set_ylabel('Percentage (%)')
    ax5.set_title('Detailed Performance by Map', fontweight='bold')
    ax5.set_xticks(x)
    ax5.set_xticklabels(maps)
    ax5.legend()
    ax5.set_ylim(0, 100)
    ax5.grid(axis='y', alpha=0.3, linestyle='--')

    plt.suptitle('Ensemble Evaluation Dashboard', fontsize=18, fontweight='bold')

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 대시보드 저장: {save_path}")
    else:
        plt.show()

    plt.close()


def visualize_all(results_file: str):
    """
    모든 시각화 생성

    앙상블 평가 결과에 대한 모든 차트를 생성합니다.
    """
    print("\n" + "=" * 60)
    print("📊 앙상블 평가 결과 시각화")
    print("=" * 60)
    print(f"결과 파일: {results_file}\n")

    # 결과 로드
    results = load_ensemble_results(results_file)

    # 파일명에서 베이스 이름 추출
    base_name = os.path.splitext(os.path.basename(results_file))[0]

    # 1. 맵별 비교 차트
    print("1️⃣ 맵별 비교 차트 생성 중...")
    plot_map_comparison(
        results,
        save_path=get_result_path(f"{base_name}_map_comparison.png")
    )

    # 2. 시드별 비교 차트
    print("2️⃣ 시드별 비교 차트 생성 중...")
    plot_seed_comparison(
        results,
        save_path=get_result_path(f"{base_name}_seed_comparison.png")
    )

    # 3. 실패 분석 차트
    print("3️⃣ 실패 분석 차트 생성 중...")
    plot_failure_analysis(
        results,
        save_path=get_result_path(f"{base_name}_failure_analysis.png")
    )

    # 4. 히트맵
    print("4️⃣ 히트맵 생성 중...")
    plot_heatmap(
        results,
        save_path=get_result_path(f"{base_name}_heatmap.png")
    )

    # 5. 종합 대시보드
    print("5️⃣ 종합 대시보드 생성 중...")
    plot_summary_dashboard(
        results,
        save_path=get_result_path(f"{base_name}_dashboard.png")
    )

    print("\n" + "=" * 60)
    print("✅ 모든 차트 생성 완료!")
    print(f"📁 저장 위치: results/")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    """
    메인 실행

    사용법:
        # 모든 차트 생성
        python visualize_ensemble.py --results ensemble_results.json

        # 특정 차트만 생성
        python visualize_ensemble.py --results ensemble_results.json --chart map
        python visualize_ensemble.py --results ensemble_results.json --chart seed
        python visualize_ensemble.py --results ensemble_results.json --chart failure
        python visualize_ensemble.py --results ensemble_results.json --chart heatmap
        python visualize_ensemble.py --results ensemble_results.json --chart dashboard
    """
    parser = argparse.ArgumentParser(description="앙상블 평가 결과 시각화")
    parser.add_argument(
        "--results",
        type=str,
        required=True,
        help="앙상블 평가 결과 JSON 파일"
    )
    parser.add_argument(
        "--chart",
        type=str,
        choices=["all", "map", "seed", "failure", "heatmap", "dashboard"],
        default="all",
        help="생성할 차트 타입"
    )

    args = parser.parse_args()

    # 결과 로드
    results = load_ensemble_results(args.results)
    base_name = os.path.splitext(os.path.basename(args.results))[0]

    # 차트 생성
    if args.chart == "all":
        visualize_all(args.results)
    elif args.chart == "map":
        plot_map_comparison(
            results,
            save_path=get_result_path(f"{base_name}_map_comparison.png")
        )
    elif args.chart == "seed":
        plot_seed_comparison(
            results,
            save_path=get_result_path(f"{base_name}_seed_comparison.png")
        )
    elif args.chart == "failure":
        plot_failure_analysis(
            results,
            save_path=get_result_path(f"{base_name}_failure_analysis.png")
        )
    elif args.chart == "heatmap":
        plot_heatmap(
            results,
            save_path=get_result_path(f"{base_name}_heatmap.png")
        )
    elif args.chart == "dashboard":
        plot_summary_dashboard(
            results,
            save_path=get_result_path(f"{base_name}_dashboard.png")
        )
