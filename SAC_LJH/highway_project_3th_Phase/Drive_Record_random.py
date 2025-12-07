"""
주행 영상 GIF 녹화 스크립트

모델의 주행을 GIF로 녹화하여 시각적으로 확인
"""

import os
from metadrive import MetaDriveEnv
from stable_baselines3 import PPO, SAC, TD3


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
        # 기본값: PPO
        return PPO.load(model_path)


def record_driving_gif(model_path,
                       seed=1000,
                       output_name="driving.gif",
                       max_steps=1000,
                       map_string=None):
    """
    모델의 주행을 GIF로 녹화

    Args:
        model_path: 모델 경로
        seed: 시드
        output_name: 출력 GIF 파일명 (전체 경로 포함 가능)
        max_steps: 최대 스텝 수
        map_string: MetaDrive 맵 문자열 (예: 'TSCO', 'SSSS' 등, None이면 기본 설정 사용)

    Returns:
        dict: 주행 결과 정보
            {
                "seed": seed,
                "success": bool,
                "crash": bool,
                "out_of_road": bool,
                "total_reward": float,
                "steps": int,
                "gif_path": str
            }
    """
    from config import FIXED_SEED_ENV_CONFIG

    print("\n" + "=" * 60)
    print("🎬 주행 영상 녹화 시작")
    print("=" * 60)
    print(f"모델: {model_path}")
    print(f"시드: {seed}")
    if map_string is not None:
        print(f"맵: {map_string}")
    print(f"출력: {output_name}")
    print("=" * 60 + "\n")

    # 환경 설정
    env_config = FIXED_SEED_ENV_CONFIG.copy()

    # 🔹 맵 문자열이 지정된 경우 해당 맵 사용
    if map_string is not None:
        env_config["map"] = map_string

    env_config["start_seed"] = seed
    env_config["num_scenarios"] = 1
    env_config["use_render"] = False  # 3D 렌더링 끄기 (GIF만 필요)

    # 환경 및 모델 로드
    env = MetaDriveEnv(env_config)
    obs, info = env.reset(seed=seed)
    model = load_model(model_path)

    # 에피소드 실행 및 녹화
    done = False
    total_reward = 0.0
    steps = 0

    print("🚗 주행 중...\n")

    try:
        while not done and steps < max_steps:
            # 액션 예측
            action, _ = model.predict(obs, deterministic=True)

            # 스텝 실행
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            steps += 1

            # 탑다운 뷰 렌더링 및 녹화
            env.render(
                mode="topdown",
                window=False,            # 창 표시 안 함
                screen_record=True,      # 화면 녹화 활성화
                screen_size=(1024, 1080),
                camera_position=(50, 50),
                scaling=2,
            )

            # 진행 상황 출력 (100스텝마다)
            if steps % 100 == 0:
                print(f"  Step {steps}/{max_steps} - Reward: {total_reward:.1f}")

        # GIF 생성
        print(f"\n📹 GIF 생성 중...")
        env.top_down_renderer.generate_gif(gif_name=output_name)

        # 결과 정보 정리
        success = bool(info.get("arrive_dest", False))
        crash = bool(info.get("crash", False))
        out_of_road = bool(info.get("out_of_road", False))

        # 결과 출력
        print("\n" + "=" * 60)
        print("✅ 녹화 완료!")
        print("=" * 60)
        print(f"총 보상: {total_reward:.2f}")
        print(f"스텝 수: {steps}")
        print(f"성공: {success}")
        print(f"충돌: {crash}")
        print(f"도로 이탈: {out_of_road}")
        print(f"\n💾 GIF 저장: {output_name}")
        print("=" * 60 + "\n")

        return {
            "seed": seed,
            "success": success,
            "crash": crash,
            "out_of_road": out_of_road,
            "total_reward": float(total_reward),
            "steps": int(steps),
            "gif_path": output_name,
        }

    finally:
        env.close()


def record_multiple_episodes(model_path,
                             seed,
                             n_episodes=3,
                             output_dir="gifs",
                             map_string=None):
    """
    여러 에피소드를 각각 GIF로 녹화

    (파일명은 seed_ep{n}.gif 형태, 성공/실패 정보는 콘솔에서만 확인)

    Args:
        model_path: 모델 경로
        seed: 시드
        n_episodes: 녹화할 에피소드 수
        output_dir: 출력 디렉토리
        map_string: MetaDrive 맵 문자열 (옵션)
    """
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print(f"🎬 {n_episodes}개 에피소드 녹화")
    print("=" * 60 + "\n")

    for ep in range(n_episodes):
        print(f"\n{'=' * 60}")
        print(f"에피소드 {ep + 1}/{n_episodes}")
        print(f"{'=' * 60}")

        output_name = os.path.join(output_dir, f"seed_{seed}_ep{ep + 1}.gif")

        record_driving_gif(
            model_path=model_path,
            seed=seed,
            output_name=output_name,
            max_steps=1000,
            map_string=map_string,
        )

    print(f"\n✅ 모든 에피소드 녹화 완료!")
    print(f"📁 저장 위치: {output_dir}/\n")


def record_comparison(model_path,
                      seeds,
                      output_name="comparison.gif",
                      map_string=None):
    """
    여러 시드 비교 (단, 각각 개별 GIF로 저장)

    Args:
        model_path: 모델 경로
        seeds: 시드 리스트
        output_name: 출력 파일명 (사용 안 함, 개별 저장)
        map_string: MetaDrive 맵 문자열 (옵션)
    """
    output_dir = "gifs/comparison"
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print(f"🎬 {len(seeds)}개 시드 비교 녹화")
    print("=" * 60 + "\n")

    for i, seed in enumerate(seeds):
        print(f"\n{'=' * 60}")
        print(f"시드 {seed} ({i + 1}/{len(seeds)})")
        print(f"{'=' * 60}")

        output_name = os.path.join(output_dir, f"seed_{seed}.gif")

        record_driving_gif(
            model_path=model_path,
            seed=seed,
            output_name=output_name,
            max_steps=1000,
            map_string=map_string,
        )

    print(f"\n✅ 모든 시드 녹화 완료!")
    print(f"📁 저장 위치: {output_dir}/\n")


def record_success_and_failure(model_path,
                               success_seed,
                               failure_seed,
                               map_string=None):
    """
    성공 사례와 실패 사례를 각각 녹화

    Args:
        model_path: 모델 경로
        success_seed: 성공하는 시드
        failure_seed: 실패하는 시드
        map_string: MetaDrive 맵 문자열 (옵션)
    """
    output_dir = "gifs/comparison"
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print("🎬 성공 vs 실패 비교 녹화")
    print("=" * 60 + "\n")

    # 성공 사례
    print(f"\n{'=' * 60}")
    print("✅ 성공 사례 녹화")
    print(f"{'=' * 60}")

    record_driving_gif(
        model_path=model_path,
        seed=success_seed,
        output_name=os.path.join(output_dir, "success.gif"),
        max_steps=1000,
        map_string=map_string,
    )

    # 실패 사례
    print(f"\n{'=' * 60}")
    print("❌ 실패 사례 녹화")
    print(f"{'=' * 60}")

    record_driving_gif(
        model_path=model_path,
        seed=failure_seed,
        output_name=os.path.join(output_dir, "failure.gif"),
        max_steps=1000,
        map_string=map_string,
    )

    print(f"\n✅ 비교 녹화 완료!")
    print(f"📁 저장 위치: {output_dir}/")
    print(f"   - success.gif")
    print(f"   - failure.gif\n")


# ✅ TEST_SEEDS 전체에 대해 자동 녹화
def record_all_test_seeds_with_status(model_path,
                                      seeds,
                                      n_episodes=1,
                                      base_dir="gifs/auto",
                                      max_steps=1000,
                                      map_string=None):
    """
    TEST_SEEDS 등 여러 시드에 대해,
    - 각 map_string별 상위 폴더 생성 (예: gifs/auto/TSCO/)
    - 그 아래에 seed별 폴더 생성 (예: gifs/auto/TSCO/seed_9935/)
    - 각 에피소드 GIF를 success/fail 접미사로 저장

    예: gifs/auto/TSCO/seed_9935/seed9935_ep1_success.gif

    Args:
        model_path: 모델 경로
        seeds: 시드 리스트
        n_episodes: 시드당 에피소드 수
        base_dir: 최상위 출력 디렉토리
        max_steps: 최대 스텝 수
        map_string: 맵 문자열 (폴더 구조 및 env_config에 모두 반영)
    """
    # map_string이 있다면 base_dir/map_string 구조 사용
    if map_string is not None:
        map_dir = os.path.join(base_dir, map_string)
    else:
        map_dir = base_dir

    os.makedirs(map_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print("🎬 모든 시드 자동 녹화 (success/fail 파일명 구분)")
    print("=" * 60)
    print(f"시드 목록: {seeds}")
    print(f"에피소드/시드: {n_episodes}")
    print(f"출력 디렉토리: {map_dir}")
    if map_string is not None:
        print(f"맵: {map_string}")
    print("=" * 60 + "\n")

    for seed in seeds:
        seed_dir = os.path.join(map_dir, f"seed_{seed}")
        os.makedirs(seed_dir, exist_ok=True)

        print(f"\n{'=' * 60}")
        print(f"Seed {seed} 처리 시작 (저장 폴더: {seed_dir})")
        print(f"{'=' * 60}")

        for ep in range(n_episodes):
            ep_idx = ep + 1
            tmp_name = os.path.join(seed_dir, f"tmp_seed{seed}_ep{ep_idx}.gif")

            # 1) 임시 이름으로 녹화
            result = record_driving_gif(
                model_path=model_path,
                seed=seed,
                output_name=tmp_name,
                max_steps=max_steps,
                map_string=map_string,
            )

            # 2) 성공/실패 상태에 따라 파일명 변경
            status = "success" if result["success"] else "fail"
            final_name = os.path.join(
                seed_dir,
                f"seed{seed}_ep{ep_idx}_{status}.gif",
            )

            # tmp -> final rename
            try:
                os.replace(tmp_name, final_name)
            except FileNotFoundError:
                # 혹시라도 생성이 안 되었다면 경고만 출력
                print(f"⚠️ GIF 파일을 찾을 수 없어 이름 변경 실패: {tmp_name}")
            else:
                print(f"💾 최종 저장: {final_name}")

    print("\n" + "=" * 60)
    print("✅ 모든 시드 자동 녹화 완료!")
    print(f"📁 최상위 디렉토리: {map_dir}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    """
    사용법:
        # 단일 에피소드 녹화
        python Drive_Record_random.py --model models/td3_tsco_map_500k.zip --seed 1000 --map TSCO

        # 여러 에피소드 녹화
        python Drive_Record_random.py --model models/td3_tsco_map_500k.zip --seed 2679 --episodes 3 --map TSCO

        # 여러 시드 비교
        python Drive_Record_random.py --model models/td3_tsco_map_500k.zip --seeds 1000 2679 4657 --compare --map TSCO

        # 성공 vs 실패 비교
        python Drive_Record_random.py --model models/td3_tsco_map_500k.zip --success 1000 --failure 2679 --map TSCO

        # ✅ TEST_SEEDS 전체에 대해 자동 녹화 (map별 폴더 + seed별 하위 폴더)
        python Drive_Record_random.py --model models/td3_tsco_map_500k.zip --all-test-seeds --episodes 3 --map TSCO
    """
    import argparse
    from config import TEST_SEEDS

    parser = argparse.ArgumentParser(description="주행 영상 GIF 녹화")
    parser.add_argument("--model", type=str, required=True,
                        help="모델 파일 경로")
    parser.add_argument("--seed", type=int, default=None,
                        help="단일 시드")
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                        help="여러 시드")
    parser.add_argument("--episodes", type=int, default=1,
                        help="에피소드 수")
    parser.add_argument("--output", type=str, default="driving.gif",
                        help="출력 파일명 (단일 에피소드용)")
    parser.add_argument("--max-steps", type=int, default=1000,
                        help="최대 스텝 수")
    parser.add_argument("--compare", action="store_true",
                        help="여러 시드 비교 모드")
    parser.add_argument("--success", type=int, default=None,
                        help="성공 시드 (--failure와 함께 사용)")
    parser.add_argument("--failure", type=int, default=None,
                        help="실패 시드 (--success와 함께 사용)")
    parser.add_argument("--all-test-seeds", action="store_true",
                        help="config.TEST_SEEDS 전체에 대해 "
                             "시드별 폴더 + success/fail 파일명으로 자동 녹화")
    parser.add_argument("--map", type=str, default=None,
                        help="MetaDrive 맵 문자열 (예: TSCO, SSSS 등)")

    args = parser.parse_args()

    # 1) 성공 vs 실패 비교
    if args.success is not None and args.failure is not None:
        record_success_and_failure(
            model_path=args.model,
            success_seed=args.success,
            failure_seed=args.failure,
            map_string=args.map,
        )

    # 2) TEST_SEEDS 전체 자동 녹화 (map별 폴더 + seed별 폴더)
    elif args.all_test_seeds:
        record_all_test_seeds_with_status(
            model_path=args.model,
            seeds=TEST_SEEDS,
            n_episodes=args.episodes,
            base_dir="gifs/auto",
            max_steps=args.max_steps,
            map_string=args.map,
        )

    # 3) 여러 시드 비교 (개별 GIF)
    elif args.compare and args.seeds:
        record_comparison(
            model_path=args.model,
            seeds=args.seeds,
            map_string=args.map,
        )

    # 4) 하나의 시드에 대해 여러 에피소드 녹화
    elif args.seed and args.episodes > 1:
        record_multiple_episodes(
            model_path=args.model,
            seed=args.seed,
            n_episodes=args.episodes,
            output_dir="gifs",
            map_string=args.map,
        )

    # 5) 단일 에피소드
    elif args.seed:
        record_driving_gif(
            model_path=args.model,
            seed=args.seed,
            output_name=args.output,
            max_steps=args.max_steps,
            map_string=args.map,
        )

    # 6) 시드 리스트 개별 녹화 (기존 방식)
    elif args.seeds:
        for seed in args.seeds:
            output_name = f"seed_{seed}.gif"
            record_driving_gif(
                model_path=args.model,
                seed=seed,
                output_name=output_name,
                max_steps=args.max_steps,
                map_string=args.map,
            )

    # 7) 어떤 시드 옵션도 없을 때 안내
    else:
        print("❌ --seed, --seeds, --success/--failure 또는 --all-test-seeds 옵션이 필요합니다")
        print("예시:")
        print("  python Drive_Record_random.py --model MODEL_PATH --all-test-seeds --episodes 3 --map TSCO")
