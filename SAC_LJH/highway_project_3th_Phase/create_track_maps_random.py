"""
랜덤 맵 평가 결과 기반 TRACO 시각화 스크립트

- 입력: evaluate_random_maps.py가 생성한 random_maps_evaluation.json
- 출력: 맵 타입(map_string)별로, 시드별 궤적+맵(TRACO) 이미지

예시:
    # 1) 모든 맵에 대해 TRACO 이미지 생성
    python create_track_maps_random.py \
        --results results/random_maps_evaluation.json \
        --model models/td3_tsco_map_500k.zip \
        --save-dir traco_random

    # 2) 성공률 상위 3개 맵만 생성
    python create_track_maps_random.py \
        --results results/random_maps_evaluation.json \
        --model models/td3_tsco_map_500k.zip \
        --top-k 3 \
        --save-dir traco_random_top3

    # 3) 특정 맵(TSCO, STCO)만 생성
    python create_track_maps_random.py \
        --results results/random_maps_evaluation.json \
        --model models/td3_tsco_map_500k.zip \
        --maps TSCO STCO
"""

import sys
import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from metadrive import MetaDriveEnv
from stable_baselines3 import PPO, SAC, TD3

# 프로젝트 루트에서 모듈 찾도록 경로 추가
sys.path.insert(0, '.')

from utils.path_utils import get_result_path
from config import FIXED_SEED_ENV_CONFIG


def load_model(model_path):
    """모델 로드"""
    model_name = os.path.basename(model_path).lower()

    if 'sac' in model_name:
        return SAC.load(model_path)
    elif 'td3' in model_name:
        return TD3.load(model_path)
    elif 'ppo' in model_name:
        return PPO.load(model_path)
    else:
        return PPO.load(model_path)


def create_traco_map_by_seed(
    map_string,
    seeds,
    model_path,
    model_name,
    n_episodes=3,
    resolution=(1024, 1024),
    save_path=None
):
    """
    여러 시드에 대해 맵 + 궤적(TRACO)을 하나의 이미지로 생성

    Args:
        map_string: MetaDrive 맵 문자열 (예: 'TSCO', 'STCO')
        seeds: 시드 리스트
        model_path: 모델 경로
        model_name: 모델 이름 (표시용)
        n_episodes: 각 시드당 에피소드 수
        resolution: 맵 해상도 (width, height)
        save_path: 저장 경로
    """
    from metadrive.utils.draw_top_down_map import draw_top_down_map

    print(f"\n🗺️  맵 {map_string} TRACO 이미지 생성 중...")
    print(f"   시드: {seeds}")
    print(f"   에피소드/시드: {n_episodes}")

    # 모델 로드
    model = load_model(model_path)

    # 각 시드별로 서브플롯 생성
    n_seeds = len(seeds)
    fig, axes = plt.subplots(1, n_seeds, figsize=(8*n_seeds, 8))

    if n_seeds == 1:
        axes = [axes]

    for idx, seed in enumerate(seeds):
        print(f"\n   처리 중: 시드 {seed}")

        # 환경 설정
        env_config = FIXED_SEED_ENV_CONFIG.copy()
        env_config["map"] = map_string
        env_config["start_seed"] = seed
        env_config["num_scenarios"] = 1
        env_config["use_render"] = False
        env_config["log_level"] = 50

        # 환경 생성
        env = MetaDriveEnv(env_config)

        # 모든 궤적 수집 (궤적 + 성공 여부)
        all_trajectories = []

        for ep in range(n_episodes):
            obs, info = env.reset(seed=seed)
            trajectory = []
            done = False
            steps = 0
            max_steps = 1000

            while not done and steps < max_steps:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                # 차량 위치 기록
                if hasattr(env.vehicle, 'position'):
                    pos = env.vehicle.position
                    trajectory.append([pos[0], pos[1]])

                steps += 1

            # 성공 여부 확인
            success = info.get('arrive_dest', False)
            status_str = "✅ 성공" if success else "❌ 실패"

            all_trajectories.append({
                'trajectory': np.array(trajectory),
                'success': success
            })
            print(f"      에피소드 {ep+1}/{n_episodes}: {steps} 스텝 - {status_str}")

        # 맵 이미지 그리기
        map_image = draw_top_down_map(env.current_map, resolution=resolution)

        # 서브플롯에 맵 표시
        ax = axes[idx]
        ax.imshow(map_image, cmap='bone', origin='lower')

        # 궤적 그리기
        img_height, img_width = map_image.shape[:2]

        # 모든 궤적의 좌표 범위 계산 (정규화를 위해)
        all_x = []
        all_y = []
        for traj_info in all_trajectories:
            traj = traj_info['trajectory']
            if len(traj) > 0:
                all_x.extend(traj[:, 0])
                all_y.extend(traj[:, 1])

        if len(all_x) > 0:
            min_x = np.min(all_x)
            max_x = np.max(all_x)
            min_y = np.min(all_y)
            max_y = np.max(all_y)

            for traj_info in all_trajectories:
                traj = traj_info['trajectory']
                success = traj_info['success']

                if len(traj) > 0:
                    # 좌표 변환: 월드 좌표 -> 이미지 픽셀 좌표
                    # visualize_trajectory.py 방식 사용
                    traj_x = (traj[:, 0] - min_x) / (max_x - min_x + 1e-6) * img_width
                    traj_y = (traj[:, 1] - min_y) / (max_y - min_y + 1e-6) * img_height
                    # Y축 반전 (이미지 좌표계)
                    traj_y = img_height - traj_y

                    # 성공: 초록색, 실패: 빨간색
                    color = 'green' if success else 'red'
                    ax.plot(traj_x, traj_y, color=color, linewidth=2, alpha=0.7)

                    # 시작점: 파란색 동그라미
                    ax.plot(traj_x[0], traj_y[0], 'bo', markersize=10,
                           markeredgecolor='white', markeredgewidth=2)

                    # 끝점: 마커 (성공: 별, 실패: X)
                    if success:
                        ax.plot(traj_x[-1], traj_y[-1], 'g*', markersize=15,
                               markeredgecolor='white', markeredgewidth=1.5)
                    else:
                        ax.plot(traj_x[-1], traj_y[-1], 'rX', markersize=12,
                               markeredgecolor='white', markeredgewidth=1.5)

        ax.set_title(f'Seed {seed}', fontsize=14, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])

        env.close()

    plt.suptitle(f'Map: {map_string} | Model: {model_name}',
                 fontsize=16, fontweight='bold', y=0.98)

    # 범례 추가
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='blue',
               markersize=10, label='Start Point 🔵'),
        Line2D([0], [0], color='green', linewidth=2, label='Success Path ✅'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor='green',
               markersize=12, label='Success End ⭐'),
        Line2D([0], [0], color='red', linewidth=2, label='Failure Path ❌'),
        Line2D([0], [0], marker='X', color='w', markerfacecolor='red',
               markersize=10, label='Failure End ✖️'),
    ]
    fig.legend(handles=legend_elements, loc='upper right', fontsize=11, framealpha=0.9)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n💾 저장 완료: {save_path}")
    else:
        plt.show()

    plt.close()


def load_random_map_results(results_path: str) -> dict:
    """evaluate_random_maps.py 결과 JSON 로드"""
    if not os.path.exists(results_path):
        raise FileNotFoundError(f"결과 파일을 찾을 수 없습니다: {results_path}")

    with open(results_path, "r", encoding="utf-8") as f:
        results = json.load(f)

    if "map_results" not in results:
        raise KeyError("results JSON에 'map_results' 키가 없습니다. "
                       "evaluate_random_maps.py 결과인지 확인해 주세요.")

    return results


def select_maps(results: dict, target_maps=None, top_k: int | None = None):
    """
    시각화 대상 맵 선택

    Args:
        results: random_maps_evaluation.json 로드 결과
        target_maps: 사용자가 직접 지정한 맵 리스트 (예: ['TSCO', 'STCO'])
        top_k: 성공률 상위 k개 맵만 선택 (target_maps가 없을 때만 의미 있음)

    Returns:
        list[str]: 선택된 map_string 리스트
    """
    map_results = results["map_results"]
    all_maps = list(map_results.keys())

    # 1) 특정 맵이 명시된 경우: 교집합만 사용
    if target_maps:
        selected = [m for m in target_maps if m in map_results]
        missing = sorted(set(target_maps) - set(selected))
        if missing:
            print(f"⚠️ 결과에 존재하지 않는 맵이 있어 제외되었습니다: {missing}")
        if not selected:
            raise ValueError("유효한 맵이 선택되지 않았습니다.")
        return selected

    # 2) 그렇지 않으면 성공률 기준 정렬 후 top_k 사용
    #    summary["overall_success_rate"] 기준으로 내림차순 정렬
    map_success = []
    for m, r in map_results.items():
        s = r.get("summary", {}).get("overall_success_rate", None)
        if s is None:
            print(f"⚠️ 맵 {m} 에 overall_success_rate가 없어 제외합니다.")
            continue
        map_success.append((m, float(s)))

    if not map_success:
        raise ValueError("summary.overall_success_rate를 가진 맵이 없습니다.")

    # 성공률 내림차순 정렬
    map_success.sort(key=lambda x: x[1], reverse=True)
    ordered_maps = [m for m, _ in map_success]

    if top_k is not None and top_k < len(ordered_maps):
        return ordered_maps[:top_k]
    return ordered_maps


def main():
    parser = argparse.ArgumentParser(
        description="랜덤 맵 평가 결과 기반 TRACO 맵 생성 (맵별 시드 궤적 이미지)"
    )
    parser.add_argument(
        "--results",
        type=str,
        required=True,
        help="evaluate_random_maps.py로 생성한 JSON 결과 파일 경로 "
             "(예: results/random_maps_evaluation.json)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="모델 경로 (예: models/td3_tsco_map_500k.zip)"
    )
    parser.add_argument(
        "--maps",
        type=str,
        nargs="+",
        default=None,
        help="시각화할 맵 문자열 리스트 (예: TSCO STCO). "
             "생략 시 성공률 기준으로 자동 선택"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="성공률 상위 K개의 맵만 시각화 (maps 미지정 시에만 적용)"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=3,
        help="시드별 실행 에피소드 수 (기본: 3)"
    )
    parser.add_argument(
        "--resolution",
        type=int,
        nargs=2,
        default=[1024, 1024],
        help="맵 해상도 [width height] (기본: 1024 1024)"
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default=None,
        help="출력 이미지를 저장할 디렉토리 "
             "(기본: results/traco_random_maps/ 아래에 생성)"
    )

    args = parser.parse_args()

    # 1) 결과 로드
    results = load_random_map_results(args.results)
    map_results = results["map_results"]
    model_path = args.model

    # 2) 시각화 대상 맵 선택
    selected_maps = select_maps(
        results,
        target_maps=args.maps,
        top_k=args.top_k
    )

    print("\n" + "=" * 60)
    print("🎯 랜덤 맵 TRACO 시각화 대상")
    print("=" * 60)
    print(f"결과 파일: {args.results}")
    print(f"총 맵 수: {len(map_results)}")
    print(f"선택된 맵 수: {len(selected_maps)}")
    print(f"맵 리스트 (순서: 성공률 기준): {selected_maps}")
    print("=" * 60 + "\n")

    # 3) 저장 디렉토리 결정
    if args.save_dir is None:
        # 결과 디렉토리 하위에 traco_random_maps/ 생성
        base_dir = get_result_path("traco_random_maps")
    else:
        base_dir = get_result_path(args.save_dir)
    os.makedirs(base_dir, exist_ok=True)

    # 4) 모델 경로 기본값 설정 (필요 시)
    if model_path is None:
        # 기존 create_traco_maps.py의 기본 로직과 유사하게,
        # TD3 → PPO 순으로 존재 여부 확인
        if os.path.exists("models/td3_tsco_map_500k.zip"):
            model_path = "models/td3_tsco_map_500k.zip"
            print(f"💡 모델 미지정: TD3 기본 모델 사용: {model_path}")
        elif os.path.exists("models/ppo_balanced_6seeds_v4_800k.zip"):
            model_path = "models/ppo_balanced_6seeds_v4_800k.zip"
            print(f"💡 모델 미지정: PPO 기본 모델 사용: {model_path}")
        else:
            raise FileNotFoundError(
                "모델 경로가 지정되지 않았고, 기본 경로에 TD3/PPO 모델이 없습니다.\n"
                "  ➜ --model 옵션으로 명시적으로 모델 경로를 지정해 주세요."
            )

    model_name = os.path.basename(model_path)

    # 5) 각 맵마다 create_traco_map_by_seed 호출
    for map_str in selected_maps:
        m_result = map_results[map_str]

        seed_results = m_result.get("seed_results", {})
        if not seed_results:
            print(f"⚠️ 맵 {map_str} 에 seed_results가 없어 건너뜁니다.")
            continue

        # JSON 키는 문자열이므로 정수 변환 후 정렬
        seeds = sorted(int(s) for s in seed_results.keys())

        # 파일명 예: traco_random_TSCO_seeds_9935_5012_....
        seed_part = "_".join(map(str, seeds))
        filename = f"traco_random_{map_str}_seeds_{seed_part}.png"
        save_path = os.path.join(base_dir, filename)

        print("\n" + "-" * 60)
        print(f"🗺️  맵 {map_str} 처리 시작")
        print(f"    시드: {seeds}")
        print(f"    저장 경로: {save_path}")
        print("-" * 60)

        # 기존 create_traco_maps.py의 함수 재사용
        # (내부에서 MetaDriveEnv / 모델 로드 / 궤적 수집 / 플롯까지 수행)
        create_traco_map_by_seed(
            map_string=map_str,
            seeds=seeds,
            model_path=model_path,
            model_name=model_name,
            n_episodes=args.episodes,
            resolution=tuple(args.resolution),
            save_path=save_path
        )

    print("\n" + "=" * 60)
    print("✅ 랜덤 맵 기반 TRACO 이미지 생성 완료")
    print(f"📁 저장 디렉토리: {base_dir}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
