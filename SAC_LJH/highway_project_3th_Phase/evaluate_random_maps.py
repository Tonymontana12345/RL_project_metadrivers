"""
랜덤 맵 조합 평가 스크립트

T, S, C, O 블록을 랜덤 순서로 조합한 평가 맵을 생성하여 모델을 테스트합니다.
"""

import itertools
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import json
import os
import argparse
import sys

from config import TEST_SEEDS, FIXED_SEED_ENV_CONFIG
from utils.path_utils import get_result_path

# evaluate.py의 함수들을 재사용
from evaluate import (
    detect_algorithm,
    load_model,
    evaluate_model,
    create_summary_table,  # ⬅ save_results는 더 이상 사용하지 않으므로 제거
)

# ✅ 그래프 생성을 위한 추가 import
import matplotlib.pyplot as plt
import seaborn as sns

# 스타일 설정 (기존 visualize.py와 동일하게)
plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")


def generate_random_map_combinations(
    blocks=["T", "S", "C", "O"], num_blocks=4, num_maps=10, seed=None
):
    """
    T, S, C, O 블록을 랜덤 순서로 조합한 맵 리스트 생성

    Args:
        blocks: 사용할 블록 리스트 (기본: ['T', 'S', 'C', 'O'])
        num_blocks: 각 맵에 사용할 블록 수
        num_maps: 생성할 맵 수
        seed: 랜덤 시드

    Returns:
        list: 맵 문자열 리스트 (예: ['TSCO', 'TOCS', 'SCTO', ...])
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # 가능한 모든 조합 생성 (중복 허용)
    # 현재는 blocks 집합에서 num_blocks 길이의 순열만 사용
    all_permutations = list(itertools.permutations(blocks, num_blocks))

    # 문자열로 변환 후 중복 제거
    unique_maps = list(set(["".join(perm) for perm in all_permutations]))

    # 요청된 수만큼 랜덤 선택 (전체보다 많으면 추가 생성)
    if num_maps <= len(unique_maps):
        selected_maps = random.sample(unique_maps, num_maps)
    else:
        selected_maps = unique_maps.copy()
        # 부족한 만큼 랜덤으로 추가 생성 (중복 허용 but 되도록 기존과 다르게)
        while len(selected_maps) < num_maps:
            map_str = "".join(random.choices(blocks, k=num_blocks))
            if map_str not in selected_maps:
                selected_maps.append(map_str)
            else:
                # 이미 있으면 블록 순서를 섞어서 추가 시도
                map_list = list(map_str)
                random.shuffle(map_list)
                new_map = "".join(map_list)
                if new_map not in selected_maps:
                    selected_maps.append(new_map)

    return selected_maps


def evaluate_on_random_maps(
    model_path,
    blocks=["T", "S", "C", "O"],
    num_blocks=4,
    num_maps=10,
    test_seeds=None,
    n_episodes=10,
    render=False,
    verbose=True,
    random_seed=None,
):
    """
    랜덤 맵 조합에서 모델 평가

    Args:
        model_path: 모델 파일 경로
        blocks: 사용할 블록 리스트
        num_blocks: 각 맵에 사용할 블록 수
        num_maps: 평가할 맵 수
        test_seeds: 테스트 시드 리스트 (None이면 TEST_SEEDS 사용)
        n_episodes: 각 맵당 에피소드 수
        render: 렌더링 여부
        verbose: 상세 출력 여부
        random_seed: 랜덤 시드

    Returns:
        dict: 평가 결과
    """
    if test_seeds is None:
        test_seeds = TEST_SEEDS

    # 랜덤 맵 조합 생성
    random_maps = generate_random_map_combinations(
        blocks=blocks,
        num_blocks=num_blocks,
        num_maps=num_maps,
        seed=random_seed,
    )

    if verbose:
        print("\n" + "=" * 60)
        print("🎲 랜덤 맵 조합 평가 시작")
        print("=" * 60)
        print(f"모델: {model_path}")
        print(f"블록: {blocks}")
        print(f"블록 수/맵: {num_blocks}")
        print(f"맵 수: {num_maps}")
        print(f"테스트 시드: {test_seeds}")
        print(f"에피소드/맵: {n_episodes}")
        print(f"\n📋 생성된 맵 조합:")
        for i, map_str in enumerate(random_maps, 1):
            print(f"   {i:2d}. {map_str}")
        print("=" * 60 + "\n")

    # 전체 결과 저장 구조
    all_results = {
        "model_path": model_path,
        "blocks": blocks,
        "num_blocks": num_blocks,
        "num_maps": num_maps,
        "test_seeds": test_seeds,
        "n_episodes": n_episodes,
        "random_maps": random_maps,
        "map_results": {},  # 각 맵별 결과
        "summary": {},  # 전체 통계
    }

    # 각 맵별로 평가
    for map_idx, map_str in enumerate(random_maps, 1):
        if verbose:
            print(f"\n{'=' * 60}")
            print(f"🗺️  맵 {map_idx}/{num_maps}: {map_str}")
            print(f"{'=' * 60}")

        # 환경 설정 (맵 지정)
        env_config = FIXED_SEED_ENV_CONFIG.copy()
        env_config["map"] = map_str

        # 각 맵에서 여러 시드로 평가 (evaluate.py의 evaluate_model 활용)
        map_results = evaluate_model(
            model_path=model_path,
            test_seeds=test_seeds,
            n_episodes=n_episodes,
            render=render,
            verbose=verbose,
            env_config=env_config,
        )

        if map_results:
            # 맵별 결과 저장 (seed_results + summary 정도만 사용)
            all_results["map_results"][map_str] = {
                "seed_results": map_results["seed_results"],
                "summary": map_results["summary"],
            }

            if verbose:
                print(f"\n📊 맵 '{map_str}' 요약:")
                print(
                    f"   평균 성공률: {map_results['summary']['overall_success_rate']*100:.1f}%"
                )
                print(
                    f"   평균 보상: {map_results['summary']['overall_mean_reward']:.2f}"
                )

    # 전체 통계 계산 (맵 summary를 기반으로 집계)
    all_success_rates = []
    all_rewards = []

    for map_str, map_result in all_results["map_results"].items():
        summary = map_result["summary"]
        all_success_rates.append(summary["overall_success_rate"])
        all_rewards.append(summary["overall_mean_reward"])

    if all_success_rates:
        all_success_rates = np.array(all_success_rates)
        all_rewards = np.array(all_rewards)

        all_results["summary"] = {
            "overall_mean_success_rate": float(np.mean(all_success_rates)),
            "overall_std_success_rate": float(np.std(all_success_rates)),
            "overall_mean_reward": float(np.mean(all_rewards)),
            "overall_std_reward": float(np.std(all_rewards)),
            "best_map": random_maps[int(np.argmax(all_success_rates))],
            "worst_map": random_maps[int(np.argmin(all_success_rates))],
        }

    if verbose and len(all_success_rates) > 0:
        print("\n" + "=" * 60)
        print("📈 전체 평가 요약")
        print("=" * 60)
        print(f"평가한 맵 수: {num_maps}")
        print(
            f"전체 평균 성공률: {all_results['summary']['overall_mean_success_rate']*100:.1f}% "
            f"± {all_results['summary']['overall_std_success_rate']*100:.1f}%"
        )
        print(
            f"전체 평균 보상: {all_results['summary']['overall_mean_reward']:.2f} "
            f"± {all_results['summary']['overall_std_reward']:.2f}"
        )
        print(f"최고 성능 맵: {all_results['summary']['best_map']}")
        print(f"최저 성능 맵: {all_results['summary']['worst_map']}")
        print("=" * 60 + "\n")

    return all_results


def create_map_comparison_table(results):
    """
    맵별 비교 표 생성

    Args:
        results: evaluate_on_random_maps 결과

    Returns:
        pd.DataFrame
    """
    data = []

    for map_str, map_result in results["map_results"].items():
        summary = map_result["summary"]
        data.append(
            {
                "Map": map_str,
                "Success Rate (%)": float(summary["overall_success_rate"] * 100),
                "Mean Reward": float(summary["overall_mean_reward"]),
                "Std Reward": float(summary["overall_std_reward"]),
            }
        )

    df = pd.DataFrame(data)
    df = df.sort_values("Success Rate (%)", ascending=False)
    return df


def save_random_map_results(results_json, filename):
    """
    랜덤 맵 평가 결과 저장용 함수
    (기존 evaluate.py의 save_results와는 별도로, 구조 그대로 저장)
    """
    save_path = get_result_path(filename)

    # 디렉토리 생성
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # JSON 통째로 저장
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(results_json, f, ensure_ascii=False, indent=2)

    print(f"💾 랜덤 맵 평가 결과 저장: {save_path}")


# ============================================================
# ✅ 새로 추가: 텍스트 리포트 & 그래프 생성
# ============================================================

def save_random_map_report(results, map_df, filename):
    """
    evaluate_random_maps.py에서 print 하던 내용을 기반으로
    텍스트 리포트(.txt)를 생성
    """
    save_path = get_result_path(filename)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    lines = []
    lines.append("=" * 60)
    lines.append("PGDrive 랜덤 맵 조합 평가 리포트")
    lines.append("=" * 60)
    lines.append("")
    lines.append(f"모델 경로       : {results.get('model_path')}")
    lines.append(f"블록 구성       : {results.get('blocks')}")
    lines.append(f"블록 수/맵      : {results.get('num_blocks')}")
    lines.append(f"평가 맵 수      : {results.get('num_maps')}")
    lines.append(f"테스트 시드     : {results.get('test_seeds')}")
    lines.append(f"에피소드/맵     : {results.get('n_episodes')}")
    lines.append("")
    lines.append("생성된 맵 조합:")
    for i, m in enumerate(results.get("random_maps", []), 1):
        lines.append(f"  {i:2d}. {m}")
    lines.append("")

    # 전체 요약
    summary = results.get("summary", {})
    if summary:
        lines.append("=" * 60)
        lines.append("전체 평가 요약")
        lines.append("=" * 60)
        lines.append(
            f"전체 평균 성공률: {summary['overall_mean_success_rate']*100:.1f}% "
            f"± {summary['overall_std_success_rate']*100:.1f}%"
        )
        lines.append(
            f"전체 평균 보상  : {summary['overall_mean_reward']:.2f} "
            f"± {summary['overall_std_reward']:.2f}"
        )
        lines.append(f"최고 성능 맵    : {summary['best_map']}")
        lines.append(f"최저 성능 맵    : {summary['worst_map']}")
        lines.append("")

    # 맵별 요약 테이블
    if map_df is not None and len(map_df) > 0:
        lines.append("=" * 60)
        lines.append("맵별 성능 비교 (성공률 내림차순)")
        lines.append("=" * 60)
        lines.append(map_df.to_string(index=False))
        lines.append("")

    # 맵별 상세 결과
    lines.append("=" * 60)
    lines.append("맵별 상세 결과")
    lines.append("=" * 60)

    for map_str, map_result in results.get("map_results", {}).items():
        ms = map_result.get("summary", {})
        seed_results = map_result.get("seed_results", {})

        lines.append(f"\n[맵 {map_str}]")
        if ms:
            lines.append(
                f"  평균 보상   : {ms['overall_mean_reward']:.2f} "
                f"± {ms['overall_std_reward']:.2f}"
            )
            lines.append(
                f"  평균 성공률 : {ms['overall_success_rate']*100:.1f}% "
                f"± {ms['success_rate_std']*100:.1f}%"
            )

        for seed in sorted(seed_results.keys(), key=lambda s: int(s)):
            st = seed_results[seed]
            lines.append(
                f"    - 시드 {seed}: "
                f"보상 {st['mean_reward']:.2f} ± {st['std_reward']:.2f}, "
                f"성공률 {st['success_rate']*100:.1f}%, "
                f"충돌 {st['crash_rate']*100:.1f}%, "
                f"도로이탈 {st['out_of_road_rate']*100:.1f}%"
            )

    with open(save_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"📝 랜덤 맵 평가 리포트 저장: {save_path}")


def plot_map_seed_results(map_str, map_result, base_name):
    """
    단일 맵에 대해 시드별 보상 / 성공률 비교 그래프 생성
    (visualize_random_maps.py에서 사용하던 레이아웃 재사용)
    """
    seed_results = map_result["seed_results"]

    seeds = []
    mean_rewards = []
    std_rewards = []
    success_rates = []

    for seed, stats in seed_results.items():
        seeds.append(int(seed))
        mean_rewards.append(stats["mean_reward"])
        std_rewards.append(stats["std_reward"])
        success_rates.append(stats["success_rate"] * 100.0)

    seeds = np.array(seeds)
    mean_rewards = np.array(mean_rewards)
    std_rewards = np.array(std_rewards)
    success_rates = np.array(success_rates)

    # 시드 순서 정렬
    order = np.argsort(seeds)
    seeds = seeds[order]
    mean_rewards = mean_rewards[order]
    std_rewards = std_rewards[order]
    success_rates = success_rates[order]

    # 그래프 생성
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    x_pos = np.arange(len(seeds))

    # 1. 보상 비교
    ax1 = axes[0]
    ax1.bar(x_pos, mean_rewards, yerr=std_rewards, capsize=5, alpha=0.7)
    ax1.set_xlabel("Seed", fontsize=12)
    ax1.set_ylabel("Mean Reward", fontsize=12)
    ax1.set_title(
        f"Map {map_str} - Reward Comparison Across Seeds",
        fontsize=14,
        fontweight="bold",
    )
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(seeds)
    ax1.grid(True, alpha=0.3, axis="y")

    # 2. 성공률 비교
    ax2 = axes[1]
    colors = [
        "red" if rate == 0 else "orange" if rate < 50 else "green"
        for rate in success_rates
    ]
    bars = ax2.bar(x_pos, success_rates, alpha=0.7, color=colors)
    ax2.set_xlabel("Seed", fontsize=12)
    ax2.set_ylabel("Success Rate (%)", fontsize=12)
    ax2.set_title(
        f"Map {map_str} - Success Rate Across Seeds",
        fontsize=14,
        fontweight="bold",
    )
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(seeds)
    ax2.set_ylim([0, 100])
    ax2.grid(True, alpha=0.3, axis="y")

    for bar, rate in zip(bars, success_rates):
        height = bar.get_height()
        y_pos = max(height + 2, 5)
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            y_pos,
            f"{rate:.1f}%",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    plt.tight_layout()

    file_name = f"{base_name}_map_{map_str}_seed_comparison.png"
    save_path = get_result_path(file_name)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"📊 맵 {map_str} 시드별 성능 그래프 저장: {save_path}")


def visualize_random_maps_from_results(results, base_name):
    """
    evaluate_random_maps.py에서 생성한 results dict를 기반으로
    맵 유형별 시드 비교 그래프를 모두 생성
    """
    map_results = results.get("map_results", {})
    if not map_results:
        print("⚠️ map_results가 비어 있어 그래프를 생성하지 않습니다.")
        return

    print("\n" + "=" * 60)
    print("📊 랜덤 맵 평가 결과 시각화 (맵별 시드 비교)")
    print("=" * 60)
    print(f"맵 수: {len(map_results)}")
    print(f"맵 리스트: {sorted(map_results.keys())}")
    print("=" * 60 + "\n")

    for map_str, map_result in map_results.items():
        print(f"🔎 맵 {map_str} 시각화 중...")
        plot_map_seed_results(map_str, map_result, base_name)

    print("\n" + "=" * 60)
    print("✅ 모든 맵 시각화 완료")
    print("=" * 60 + "\n")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    """
    메인 실행

    사용법:
        # 기본 실행 (10개 랜덤 맵 조합)
        python evaluate_random_maps.py --model models/td3_tsco_map_500k.zip

        # 맵 수 지정
        python evaluate_random_maps.py --model models/td3_tsco_map_500k.zip --num-maps 20

        # 블록 수 지정
        python evaluate_random_maps.py --model models/td3_tsco_map_500k.zip --num-blocks 5
    """
    print("### DEBUG: NEW evaluate_random_maps.py 실행 중 ###")
    parser = argparse.ArgumentParser(description="랜덤 맵 조합 평가")
    parser.add_argument(
        "--model", type=str, required=True, help="평가할 모델 경로"
    )
    parser.add_argument(
        "--blocks",
        type=str,
        nargs="+",
        default=["T", "S", "C", "O"],
        help="사용할 블록 리스트 (기본: T S C O)",
    )
    parser.add_argument(
        "--num-blocks",
        type=int,
        default=4,
        help="각 맵에 사용할 블록 수 (기본: 4)",
    )
    parser.add_argument(
        "--num-maps",
        type=int,
        default=10,
        help="평가할 맵 수 (기본: 10)",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="테스트 시드 (기본: config의 TEST_SEEDS)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=10,
        help="각 맵당 에피소드 수 (기본: 10)",
    )
    parser.add_argument(
        "--render", action="store_true", help="렌더링 활성화"
    )
    parser.add_argument(
        "--save",
        type=str,
        default="random_maps_evaluation.json",
        help="결과 저장 파일명 (기본: random_maps_evaluation.json)",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=None,
        help="랜덤 시드 (재현성 보장)",
    )

    args = parser.parse_args()

    # 평가 실행
    results = evaluate_on_random_maps(
        model_path=args.model,
        blocks=args.blocks,
        num_blocks=args.num_blocks,
        num_maps=args.num_maps,
        test_seeds=args.seeds,
        n_episodes=args.episodes,
        render=args.render,
        verbose=True,
        random_seed=args.random_seed,
    )

    if results:
        # 맵별 비교 표 출력
        df = create_map_comparison_table(results)
        print("\n📋 맵별 성능 비교 (성공률 순):")
        print(df.to_string(index=False))
        print()

        # JSON 저장
        save_random_map_results(results, args.save)

        # 🔹 base_name: json 파일명에서 .json 제거한 prefix
        base_name = os.path.splitext(args.save)[0]

        # 🔹 텍스트 리포트 저장
        report_filename = f"{base_name}_report.txt"
        save_random_map_report(results, df, report_filename)

        # 🔹 맵 유형별 시드 비교 그래프 저장
        visualize_random_maps_from_results(results, base_name)
