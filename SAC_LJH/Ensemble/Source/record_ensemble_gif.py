"""
앙상블 모델 주행 GIF 녹화 스크립트

앙상블 모델의 주행을 GIF로 녹화하여 시각적으로 확인
"""

import os
import argparse
from typing import List, Dict
from metadrive import MetaDriveEnv
from config import FIXED_SEED_ENV_CONFIG
from ensemble_agent import EnsembleAgent, load_ensemble_models


def record_ensemble_gif(
    model_paths: List[str],
    model_names: List[str] = None,
    strategy: str = "averaging",
    scenario_weights: Dict = None,
    map_str: str = "TSCO",
    seed: int = 1000,
    output_name: str = "ensemble_driving.gif",
    max_steps: int = 1000,
    verbose: bool = True
):
    """
    앙상블 모델의 주행을 GIF로 녹화

    Args:
        model_paths: 모델 파일 경로 리스트
        model_names: 모델 이름 리스트
        strategy: 앙상블 전략
        scenario_weights: 시나리오별 가중치
        map_str: 맵 문자열
        seed: 시드
        output_name: 출력 GIF 파일명
        max_steps: 최대 스텝 수
        verbose: 상세 출력 여부
    """
    if verbose:
        print("\n" + "=" * 60)
        print("🎬 앙상블 모델 주행 GIF 녹화")
        print("=" * 60)
        print(f"모델 경로: {model_paths}")
        print(f"전략: {strategy}")
        print(f"맵: {map_str}")
        print(f"시드: {seed}")
        print(f"출력: {output_name}")
        print("=" * 60 + "\n")

    # 모델 로드
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

    # 환경 설정
    import copy
    env_config = copy.deepcopy(FIXED_SEED_ENV_CONFIG)
    env_config["map"] = map_str
    env_config["start_seed"] = seed
    env_config["num_scenarios"] = 1
    env_config["use_render"] = False

    # 센서 설정을 모델 학습 시와 동일하게 (관찰 공간 91 차원)
    # 기존 vehicle_config를 완전히 덮어쓰기
    env_config["vehicle_config"] = {
        "lidar": {
            "num_lasers": 72,
            "distance": 70,
        }
    }

    # 환경 생성
    env = MetaDriveEnv(env_config)
    obs, _ = env.reset()

    # 에피소드 실행 및 녹화
    done = False
    total_reward = 0
    steps = 0

    print("🚗 주행 중...\n")

    try:
        while not done and steps < max_steps:
            # 액션 예측
            if strategy == "scenario_based":
                action, info = ensemble.predict(obs, deterministic=True, scenario=map_str)
            else:
                action, info = ensemble.predict(obs, deterministic=True)

            # 스텝 실행
            obs, reward, terminated, truncated, env_info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            steps += 1

            # 탑다운 뷰 렌더링 및 녹화
            env.render(
                mode="topdown",
                window=False,
                screen_record=True,
                screen_size=(1024, 1080),
                camera_position=(50, 50),
                scaling=2,
            )

            # 진행 상황 출력 (100스텝마다)
            if verbose and steps % 100 == 0:
                print(f"  Step {steps}/{max_steps} - Reward: {total_reward:.1f}")

        # GIF 생성
        if verbose:
            print(f"\n📹 GIF 생성 중...")
        env.top_down_renderer.generate_gif(gif_name=output_name)

        # 결과 출력
        if verbose:
            print("\n" + "=" * 60)
            print("✅ 녹화 완료!")
            print("=" * 60)
            print(f"총 보상: {total_reward:.2f}")
            print(f"스텝 수: {steps}")
            print(f"성공: {env_info.get('arrive_dest', False)}")
            print(f"충돌: {env_info.get('crash', False)}")
            print(f"도로 이탈: {env_info.get('out_of_road', False)}")
            print(f"\n💾 GIF 저장: {output_name}")
            print("=" * 60 + "\n")

    finally:
        env.close()


def record_multiple_scenarios(
    model_paths: List[str],
    model_names: List[str] = None,
    strategy: str = "averaging",
    scenario_weights: Dict = None,
    maps: List[str] = None,
    seed: int = 1000,
    output_dir: str = "gifs/ensemble",
    max_steps: int = 1000
):
    """
    여러 시나리오(맵)를 각각 GIF로 녹화

    Args:
        model_paths: 모델 파일 경로 리스트
        model_names: 모델 이름 리스트
        strategy: 앙상블 전략
        scenario_weights: 시나리오별 가중치
        maps: 맵 리스트
        seed: 시드
        output_dir: 출력 디렉토리
        max_steps: 최대 스텝 수
    """
    if maps is None:
        maps = ["TSCO", "CSTO", "OSCT", "SCTO"]

    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print(f"🎬 {len(maps)}개 시나리오 녹화")
    print("=" * 60 + "\n")

    for i, map_str in enumerate(maps, 1):
        print(f"\n{'=' * 60}")
        print(f"시나리오 {i}/{len(maps)}: {map_str}")
        print(f"{'=' * 60}")

        output_name = os.path.join(
            output_dir,
            f"ensemble_{strategy}_{map_str}_seed{seed}.gif"
        )

        record_ensemble_gif(
            model_paths=model_paths,
            model_names=model_names,
            strategy=strategy,
            scenario_weights=scenario_weights,
            map_str=map_str,
            seed=seed,
            output_name=output_name,
            max_steps=max_steps,
            verbose=True
        )

    print(f"\n✅ 모든 시나리오 녹화 완료!")
    print(f"📁 저장 위치: {output_dir}/\n")


def record_strategy_comparison(
    model_paths: List[str],
    model_names: List[str] = None,
    strategies: List[str] = None,
    map_str: str = "CSTO",
    seed: int = 1000,
    output_dir: str = "gifs/ensemble/comparison",
    max_steps: int = 1000
):
    """
    여러 앙상블 전략을 비교하는 GIF 녹화

    Args:
        model_paths: 모델 파일 경로 리스트
        model_names: 모델 이름 리스트
        strategies: 앙상블 전략 리스트
        map_str: 맵 문자열
        seed: 시드
        output_dir: 출력 디렉토리
        max_steps: 최대 스텝 수
    """
    if strategies is None:
        strategies = ["averaging", "q_value_weighted", "scenario_based"]

    os.makedirs(output_dir, exist_ok=True)

    # 시나리오별 가중치
    scenario_weights = {
        "CSTO": {"TD3": 0.8, "SAC": 0.2},
        "OSCT": {"TD3": 0.3, "SAC": 0.7},
        "TSCO": {"TD3": 0.5, "SAC": 0.5},
        "SCTO": {"TD3": 0.5, "SAC": 0.5},
        "default": {"TD3": 0.5, "SAC": 0.5}
    }

    print("\n" + "=" * 60)
    print(f"🎬 {len(strategies)}개 전략 비교 녹화")
    print("=" * 60)
    print(f"맵: {map_str}, 시드: {seed}")
    print("=" * 60 + "\n")

    for i, strategy in enumerate(strategies, 1):
        print(f"\n{'=' * 60}")
        print(f"전략 {i}/{len(strategies)}: {strategy}")
        print(f"{'=' * 60}")

        output_name = os.path.join(
            output_dir,
            f"{strategy}_{map_str}_seed{seed}.gif"
        )

        record_ensemble_gif(
            model_paths=model_paths,
            model_names=model_names,
            strategy=strategy,
            scenario_weights=scenario_weights if strategy == "scenario_based" else None,
            map_str=map_str,
            seed=seed,
            output_name=output_name,
            max_steps=max_steps,
            verbose=True
        )

    print(f"\n✅ 모든 전략 녹화 완료!")
    print(f"📁 저장 위치: {output_dir}/\n")


if __name__ == "__main__":
    """
    메인 실행

    사용법:
        # 단일 시나리오 녹화
        python record_ensemble_gif.py \
            --models models/td3_tsco_map_500k.zip models/sac_tsco_map_500k.zip \
            --strategy scenario_based \
            --map CSTO \
            --seed 1000

        # 여러 시나리오 녹화
        python record_ensemble_gif.py \
            --models models/td3_tsco_map_500k.zip models/sac_tsco_map_500k.zip \
            --strategy scenario_based \
            --maps CSTO OSCT TSCO SCTO \
            --seed 1000

        # 전략 비교 녹화
        python record_ensemble_gif.py \
            --models models/td3_tsco_map_500k.zip models/sac_tsco_map_500k.zip \
            --compare-strategies \
            --map CSTO \
            --seed 1000
    """
    parser = argparse.ArgumentParser(description="앙상블 모델 주행 GIF 녹화")
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
        "--map",
        type=str,
        default="TSCO",
        help="테스트 맵"
    )
    parser.add_argument(
        "--maps",
        type=str,
        nargs="+",
        default=None,
        help="여러 맵 녹화 (--map 대신 사용)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1000,
        help="시드"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="ensemble_driving.gif",
        help="출력 GIF 파일명"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="gifs/ensemble",
        help="출력 디렉토리 (여러 맵 녹화 시)"
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=1000,
        help="최대 스텝 수"
    )
    parser.add_argument(
        "--compare-strategies",
        action="store_true",
        help="여러 전략 비교 녹화"
    )

    args = parser.parse_args()

    # 시나리오별 가중치
    scenario_weights = {
        "CSTO": {"TD3": 0.8, "SAC": 0.2},
        "OSCT": {"TD3": 0.3, "SAC": 0.7},
        "TSCO": {"TD3": 0.5, "SAC": 0.5},
        "SCTO": {"TD3": 0.5, "SAC": 0.5},
        "default": {"TD3": 0.5, "SAC": 0.5}
    }

    # 전략 비교 모드
    if args.compare_strategies:
        record_strategy_comparison(
            model_paths=args.models,
            model_names=args.model_names,
            map_str=args.map,
            seed=args.seed,
            output_dir=os.path.join(args.output_dir, "comparison"),
            max_steps=args.max_steps
        )

    # 여러 맵 녹화
    elif args.maps:
        record_multiple_scenarios(
            model_paths=args.models,
            model_names=args.model_names,
            strategy=args.strategy,
            scenario_weights=scenario_weights if args.strategy == "scenario_based" else None,
            maps=args.maps,
            seed=args.seed,
            output_dir=args.output_dir,
            max_steps=args.max_steps
        )

    # 단일 맵 녹화
    else:
        record_ensemble_gif(
            model_paths=args.models,
            model_names=args.model_names,
            strategy=args.strategy,
            scenario_weights=scenario_weights if args.strategy == "scenario_based" else None,
            map_str=args.map,
            seed=args.seed,
            output_name=args.output,
            max_steps=args.max_steps,
            verbose=True
        )
