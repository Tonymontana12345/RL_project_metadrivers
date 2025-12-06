"""
앙상블 모델 평가 스크립트

여러 RL 모델을 앙상블하여 다양한 시나리오에서 평가합니다.
"""

import argparse
import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import List, Dict

from metadrive import MetaDriveEnv
from config import TEST_SEEDS, FIXED_SEED_ENV_CONFIG
from ensemble_agent import EnsembleAgent, load_ensemble_models
from utils.path_utils import get_result_path


def evaluate_ensemble(
    model_paths: List[str],
    model_names: List[str] = None,
    strategy: str = "averaging",
    scenario_weights: Dict = None,
    test_maps: List[str] = None,
    test_seeds: List[int] = None,
    n_episodes: int = 10,
    render: bool = False,
    verbose: bool = True,
    record_gif: bool = False,
    gif_output_dir: str = "gifs/ensemble"
):
    """
    앙상블 모델 평가

    Args:
        model_paths: 모델 파일 경로 리스트
        model_names: 모델 이름 리스트
        strategy: 앙상블 전략
        scenario_weights: 시나리오별 가중치
        test_maps: 테스트 맵 리스트
        test_seeds: 테스트 시드 리스트
        n_episodes: 각 조합당 에피소드 수
        render: 렌더링 여부
        verbose: 상세 출력 여부
        record_gif: GIF 녹화 여부
        gif_output_dir: GIF 저장 디렉토리

    Returns:
        dict: 평가 결과
    """
    if test_seeds is None:
        test_seeds = TEST_SEEDS

    if test_maps is None:
        test_maps = ["TSCO", "CSTO", "OSCT", "SCTO"]

    # GIF 저장 디렉토리 생성
    if record_gif:
        os.makedirs(gif_output_dir, exist_ok=True)

    # 모델 로드
    if verbose:
        print("\n" + "=" * 60)
        print("🤝 앙상블 모델 평가 시작")
        print("=" * 60)
        print(f"모델 경로: {model_paths}")
        print(f"전략: {strategy}")
        print(f"테스트 맵: {test_maps}")
        print(f"테스트 시드: {test_seeds}")
        print(f"에피소드/조합: {n_episodes}")
        if record_gif:
            print(f"🎬 GIF 녹화: 활성화 (저장 위치: {gif_output_dir})")
        print("=" * 60 + "\n")

    print("📦 모델 로드 중...")
    models, loaded_names = load_ensemble_models(model_paths)

    if model_names is None:
        model_names = loaded_names

    # 앙상블 에이전트 생성
    ensemble = EnsembleAgent(
        models=models,
        model_names=model_names,
        strategy=strategy,
        scenario_weights=scenario_weights
    )

    # 결과 저장 구조
    results = {
        "model_paths": model_paths,
        "model_names": model_names,
        "strategy": strategy,
        "scenario_weights": scenario_weights,
        "test_maps": test_maps,
        "test_seeds": test_seeds,
        "n_episodes": n_episodes,
        "map_results": {},
        "summary": {}
    }

    # 각 맵별 평가
    for map_str in test_maps:
        if verbose:
            print(f"\n{'=' * 60}")
            print(f"🗺️  맵: {map_str}")
            print(f"{'=' * 60}")

        map_results = {
            "seed_results": {},
            "summary": {}
        }

        # 각 시드별 평가
        for seed in test_seeds:
            if verbose:
                print(f"\n🌱 시드 {seed} 평가 중...")

            seed_stats = {
                "episodes": [],
                "rewards": [],
                "successes": [],
                "crashes": [],
                "out_of_roads": []
            }

            # n_episodes 만큼 반복
            for episode in tqdm(range(n_episodes), desc=f"Map {map_str}, Seed {seed}",
                              disable=not verbose):
                # 환경 생성 - 센서 설정 단순화 (모델 학습 시와 동일하게)
                import copy
                env_config = copy.deepcopy(FIXED_SEED_ENV_CONFIG)
                env_config["map"] = map_str
                env_config["start_seed"] = seed
                env_config["use_render"] = render

                # 센서 설정을 모델 학습 시와 동일하게 (관찰 공간 91 차원)
                # 기존 vehicle_config를 완전히 덮어쓰기
                env_config["vehicle_config"] = {
                    "lidar": {
                        "num_lasers": 72,
                        "distance": 70,
                    }
                }

                env = MetaDriveEnv(env_config)
                obs, _ = env.reset()

                episode_reward = 0
                done = False
                steps = 0

                # GIF 녹화 여부 결정 (첫 번째 에피소드만)
                should_record = record_gif and episode == 0

                while not done:
                    # scenario_based 전략이면 맵 정보 전달
                    if strategy == "scenario_based":
                        action, _ = ensemble.predict(obs, deterministic=True,
                                                    scenario=map_str)
                    else:
                        action, _ = ensemble.predict(obs, deterministic=True)

                    obs, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated
                    episode_reward += reward
                    steps += 1

                    # GIF 녹화 (첫 번째 에피소드만)
                    if should_record:
                        env.render(
                            mode="topdown",
                            window=False,
                            screen_record=True,
                            screen_size=(1024, 1080),
                            camera_position=(50, 50),
                            scaling=2,
                        )

                # 결과 기록
                seed_stats["episodes"].append(episode)
                seed_stats["rewards"].append(episode_reward)
                seed_stats["successes"].append(1 if info.get("arrive_dest", False) else 0)
                seed_stats["crashes"].append(1 if info.get("crash", False) else 0)
                seed_stats["out_of_roads"].append(1 if info.get("out_of_road", False) else 0)

                # GIF 저장 (첫 번째 에피소드만)
                if should_record:
                    gif_filename = f"ensemble_{strategy}_{map_str}_seed{seed}.gif"
                    gif_path = os.path.join(gif_output_dir, gif_filename)
                    env.top_down_renderer.generate_gif(gif_name=gif_path)
                    if verbose:
                        print(f"   🎬 GIF 저장: {gif_path}")

                env.close()

            # 시드별 통계
            seed_summary = {
                "mean_reward": float(np.mean(seed_stats["rewards"])),
                "std_reward": float(np.std(seed_stats["rewards"])),
                "success_rate": float(np.mean(seed_stats["successes"])),
                "crash_rate": float(np.mean(seed_stats["crashes"])),
                "out_of_road_rate": float(np.mean(seed_stats["out_of_roads"]))
            }

            map_results["seed_results"][str(seed)] = seed_summary

            if verbose:
                print(f"   - 평균 보상: {seed_summary['mean_reward']:.2f} ± "
                      f"{seed_summary['std_reward']:.2f}")
                print(f"   - 성공률: {seed_summary['success_rate']*100:.1f}%")
                print(f"   - 충돌률: {seed_summary['crash_rate']*100:.1f}%")
                print(f"   - 도로이탈률: {seed_summary['out_of_road_rate']*100:.1f}%")

        # 맵별 전체 통계
        all_rewards = [s["mean_reward"] for s in map_results["seed_results"].values()]
        all_success_rates = [s["success_rate"] for s in map_results["seed_results"].values()]

        map_results["summary"] = {
            "overall_mean_reward": float(np.mean(all_rewards)),
            "overall_std_reward": float(np.std(all_rewards)),
            "overall_success_rate": float(np.mean(all_success_rates)),
            "success_rate_std": float(np.std(all_success_rates))
        }

        results["map_results"][map_str] = map_results

        if verbose:
            print(f"\n📊 맵 '{map_str}' 요약:")
            print(f"   평균 성공률: "
                  f"{map_results['summary']['overall_success_rate']*100:.1f}%")
            print(f"   평균 보상: "
                  f"{map_results['summary']['overall_mean_reward']:.2f}")

    # 전체 통계
    all_map_success_rates = [r["summary"]["overall_success_rate"]
                            for r in results["map_results"].values()]
    all_map_rewards = [r["summary"]["overall_mean_reward"]
                      for r in results["map_results"].values()]

    results["summary"] = {
        "overall_mean_success_rate": float(np.mean(all_map_success_rates)),
        "overall_std_success_rate": float(np.std(all_map_success_rates)),
        "overall_mean_reward": float(np.mean(all_map_rewards)),
        "overall_std_reward": float(np.std(all_map_rewards))
    }

    if verbose:
        print("\n" + "=" * 60)
        print("📈 전체 평가 요약")
        print("=" * 60)
        print(f"전체 평균 성공률: "
              f"{results['summary']['overall_mean_success_rate']*100:.1f}% ± "
              f"{results['summary']['overall_std_success_rate']*100:.1f}%")
        print(f"전체 평균 보상: "
              f"{results['summary']['overall_mean_reward']:.2f} ± "
              f"{results['summary']['overall_std_reward']:.2f}")
        print("=" * 60 + "\n")

    return results


def create_comparison_table(results: Dict) -> pd.DataFrame:
    """맵별 비교 표 생성"""
    data = []

    for map_str, map_result in results["map_results"].items():
        summary = map_result["summary"]
        data.append({
            "Map": map_str,
            "Success Rate (%)": float(summary["overall_success_rate"] * 100),
            "Mean Reward": float(summary["overall_mean_reward"]),
            "Std Reward": float(summary["overall_std_reward"])
        })

    df = pd.DataFrame(data)
    df = df.sort_values("Success Rate (%)", ascending=False)
    return df


def save_ensemble_results(results: Dict, filename: str):
    """앙상블 평가 결과 저장"""
    save_path = get_result_path(filename)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"💾 앙상블 평가 결과 저장: {save_path}")


def save_ensemble_report(results: Dict, df: pd.DataFrame, filename: str):
    """앙상블 평가 리포트 저장"""
    save_path = get_result_path(filename)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    lines = []
    lines.append("=" * 60)
    lines.append("앙상블 모델 평가 리포트")
    lines.append("=" * 60)
    lines.append("")
    lines.append(f"모델 경로: {results['model_paths']}")
    lines.append(f"모델 이름: {results['model_names']}")
    lines.append(f"앙상블 전략: {results['strategy']}")
    lines.append(f"테스트 맵: {results['test_maps']}")
    lines.append(f"테스트 시드: {results['test_seeds']}")
    lines.append(f"에피소드/조합: {results['n_episodes']}")
    lines.append("")

    # 시나리오별 가중치 (있으면)
    if results.get("scenario_weights"):
        lines.append("시나리오별 가중치:")
        for scenario, weights in results["scenario_weights"].items():
            lines.append(f"  {scenario}: {weights}")
        lines.append("")

    # 전체 요약
    summary = results["summary"]
    lines.append("=" * 60)
    lines.append("전체 평가 요약")
    lines.append("=" * 60)
    lines.append(
        f"전체 평균 성공률: {summary['overall_mean_success_rate']*100:.1f}% ± "
        f"{summary['overall_std_success_rate']*100:.1f}%"
    )
    lines.append(
        f"전체 평균 보상: {summary['overall_mean_reward']:.2f} ± "
        f"{summary['overall_std_reward']:.2f}"
    )
    lines.append("")

    # 맵별 비교 표
    lines.append("=" * 60)
    lines.append("맵별 성능 비교 (성공률 내림차순)")
    lines.append("=" * 60)
    lines.append(df.to_string(index=False))
    lines.append("")

    # 맵별 상세 결과
    lines.append("=" * 60)
    lines.append("맵별 상세 결과")
    lines.append("=" * 60)

    for map_str, map_result in results["map_results"].items():
        ms = map_result["summary"]
        seed_results = map_result["seed_results"]

        lines.append(f"\n[맵 {map_str}]")
        lines.append(
            f"  평균 보상: {ms['overall_mean_reward']:.2f} ± "
            f"{ms['overall_std_reward']:.2f}"
        )
        lines.append(
            f"  평균 성공률: {ms['overall_success_rate']*100:.1f}% ± "
            f"{ms['success_rate_std']*100:.1f}%"
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

    print(f"📝 앙상블 평가 리포트 저장: {save_path}")


if __name__ == "__main__":
    """
    메인 실행

    사용법:
        # 기본 실행 (평균 전략)
        python evaluate_ensemble.py \
            --models models/td3_tsco_map_500k.zip models/sac_tsco_map_500k.zip

        # 시나리오 기반 전략
        python evaluate_ensemble.py \
            --models models/td3_tsco_map_500k.zip models/sac_tsco_map_500k.zip \
            --strategy scenario_based

        # Q-value 가중 전략
        python evaluate_ensemble.py \
            --models models/td3_tsco_map_500k.zip models/sac_tsco_map_500k.zip \
            --strategy q_value_weighted
    """
    parser = argparse.ArgumentParser(description="앙상블 모델 평가")
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        required=True,
        help="앙상블할 모델 경로 리스트"
    )
    parser.add_argument(
        "--model-names",
        type=str,
        nargs="+",
        default=None,
        help="모델 이름 리스트 (예: TD3 SAC)"
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="averaging",
        choices=["voting", "averaging", "q_value_weighted",
                "scenario_based", "confidence_weighted"],
        help="앙상블 전략"
    )
    parser.add_argument(
        "--maps",
        type=str,
        nargs="+",
        default=["TSCO", "CSTO", "OSCT", "SCTO"],
        help="테스트 맵 리스트"
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="테스트 시드 리스트"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=10,
        help="각 조합당 에피소드 수"
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="렌더링 활성화"
    )
    parser.add_argument(
        "--save",
        type=str,
        default="ensemble_evaluation.json",
        help="결과 저장 파일명"
    )
    parser.add_argument(
        "--record-gif",
        action="store_true",
        help="GIF 녹화 활성화 (각 맵의 첫 번째 에피소드만)"
    )
    parser.add_argument(
        "--gif-dir",
        type=str,
        default="gifs/ensemble",
        help="GIF 저장 디렉토리 (기본: gifs/ensemble)"
    )

    args = parser.parse_args()

    # 시나리오별 가중치 (scenario_based 전략용)
    scenario_weights = {
        "CSTO": {"TD3": 0.8, "SAC": 0.2},  # TD3가 CSTO에서 강함
        "OSCT": {"TD3": 0.3, "SAC": 0.7},  # SAC가 OSCT에서 그나마 나음
        "TSCO": {"TD3": 0.5, "SAC": 0.5},
        "SCTO": {"TD3": 0.5, "SAC": 0.5},
        "default": {"TD3": 0.5, "SAC": 0.5}
    }

    # 평가 실행
    results = evaluate_ensemble(
        model_paths=args.models,
        model_names=args.model_names,
        strategy=args.strategy,
        scenario_weights=scenario_weights if args.strategy == "scenario_based" else None,
        test_maps=args.maps,
        test_seeds=args.seeds,
        n_episodes=args.episodes,
        render=args.render,
        verbose=True,
        record_gif=args.record_gif,
        gif_output_dir=args.gif_dir
    )

    if results:
        # 맵별 비교 표
        df = create_comparison_table(results)
        print("\n📋 맵별 성능 비교 (성공률 순):")
        print(df.to_string(index=False))
        print()

        # JSON 저장
        save_ensemble_results(results, args.save)

        # 리포트 저장
        base_name = os.path.splitext(args.save)[0]
        report_filename = f"{base_name}_report.txt"
        save_ensemble_report(results, df, report_filename)
