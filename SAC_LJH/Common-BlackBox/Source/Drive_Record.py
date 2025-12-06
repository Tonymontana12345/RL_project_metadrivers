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
        return PPO.load(model_path)


def record_driving_gif(model_path, seed=1000, output_name="driving.gif", max_steps=1000):
    """
    모델의 주행을 GIF로 녹화
    
    Args:
        model_path: 모델 경로
        seed: 시드
        output_name: 출력 GIF 파일명
        max_steps: 최대 스텝 수
    """
    from config import FIXED_SEED_ENV_CONFIG
    
    print("\n" + "="*60)
    print("🎬 주행 영상 녹화 시작")
    print("="*60)
    print(f"모델: {model_path}")
    print(f"시드: {seed}")
    print(f"출력: {output_name}")
    print("="*60 + "\n")
    
    # 환경 설정
    env_config = FIXED_SEED_ENV_CONFIG.copy()
    env_config["start_seed"] = seed
    env_config["num_scenarios"] = 1
    env_config["use_render"] = False  # 3D 렌더링 끄기 (GIF만 필요)
    
    # 환경 및 모델 로드
    env = MetaDriveEnv(env_config)
    model = load_model(model_path)
    
    # 에피소드 실행 및 녹화
    obs, info = env.reset(seed=seed)
    done = False
    total_reward = 0
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
                window=False,           # 창 표시 안 함
                screen_record=True,     # 화면 녹화 활성화
                screen_size=(1024, 1080), # 화면 크기
                camera_position=(50, 50),  # 카메라 위치
                scaling=2,              # 확대/축소
            )
            
            # 진행 상황 출력 (100스텝마다)
            if steps % 100 == 0:
                print(f"  Step {steps}/{max_steps} - Reward: {total_reward:.1f}")
        
        # GIF 생성
        print(f"\n📹 GIF 생성 중...")
        env.top_down_renderer.generate_gif(gif_name=output_name)
        
        # 결과 출력
        print("\n" + "="*60)
        print("✅ 녹화 완료!")
        print("="*60)
        print(f"총 보상: {total_reward:.2f}")
        print(f"스텝 수: {steps}")
        print(f"성공: {info.get('arrive_dest', False)}")
        print(f"충돌: {info.get('crash', False)}")
        print(f"도로 이탈: {info.get('out_of_road', False)}")
        print(f"\n💾 GIF 저장: {output_name}")
        print("="*60 + "\n")
    
    finally:
        env.close()


def record_multiple_episodes(model_path, seed, n_episodes=3, output_dir="gifs"):
    """
    여러 에피소드를 각각 GIF로 녹화
    
    Args:
        model_path: 모델 경로
        seed: 시드
        n_episodes: 녹화할 에피소드 수
        output_dir: 출력 디렉토리
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*60)
    print(f"🎬 {n_episodes}개 에피소드 녹화")
    print("="*60 + "\n")
    
    for ep in range(n_episodes):
        print(f"\n{'='*60}")
        print(f"에피소드 {ep + 1}/{n_episodes}")
        print(f"{'='*60}")
        
        output_name = os.path.join(output_dir, f"seed_{seed}_ep{ep+1}.gif")
        
        record_driving_gif(
            model_path=model_path,
            seed=seed,
            output_name=output_name,
            max_steps=1000
        )
    
    print(f"\n✅ 모든 에피소드 녹화 완료!")
    print(f"📁 저장 위치: {output_dir}/\n")


def record_comparison(model_path, seeds, output_name="comparison.gif"):
    """
    여러 시드 비교 (단, 각각 개별 GIF로 저장)
    
    Args:
        model_path: 모델 경로
        seeds: 시드 리스트
        output_name: 출력 파일명 (사용 안 함, 개별 저장)
    """
    output_dir = "gifs/comparison"
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*60)
    print(f"🎬 {len(seeds)}개 시드 비교 녹화")
    print("="*60 + "\n")
    
    for i, seed in enumerate(seeds):
        print(f"\n{'='*60}")
        print(f"시드 {seed} ({i+1}/{len(seeds)})")
        print(f"{'='*60}")
        
        output_name = os.path.join(output_dir, f"seed_{seed}.gif")
        
        record_driving_gif(
            model_path=model_path,
            seed=seed,
            output_name=output_name,
            max_steps=1000
        )
    
    print(f"\n✅ 모든 시드 녹화 완료!")
    print(f"📁 저장 위치: {output_dir}/\n")


def record_success_and_failure(model_path, success_seed, failure_seed):
    """
    성공 사례와 실패 사례를 각각 녹화
    
    Args:
        model_path: 모델 경로
        success_seed: 성공하는 시드
        failure_seed: 실패하는 시드
    """
    output_dir = "gifs/comparison"
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*60)
    print("🎬 성공 vs 실패 비교 녹화")
    print("="*60 + "\n")
    
    # 성공 사례
    print(f"\n{'='*60}")
    print("✅ 성공 사례 녹화")
    print(f"{'='*60}")
    
    record_driving_gif(
        model_path=model_path,
        seed=success_seed,
        output_name=os.path.join(output_dir, "success.gif"),
        max_steps=1000
    )
    
    # 실패 사례
    print(f"\n{'='*60}")
    print("❌ 실패 사례 녹화")
    print(f"{'='*60}")
    
    record_driving_gif(
        model_path=model_path,
        seed=failure_seed,
        output_name=os.path.join(output_dir, "failure.gif"),
        max_steps=1000
    )
    
    print(f"\n✅ 비교 녹화 완료!")
    print(f"📁 저장 위치: {output_dir}/")
    print(f"   - success.gif")
    print(f"   - failure.gif\n")


if __name__ == "__main__":
    """
    사용법:
        # 단일 에피소드 녹화
        python record_driving_gif.py --model models/sac_fixed_seed_1000.zip --seed 1000
        
        # 여러 에피소드 녹화
        python record_driving_gif.py --model models/sac_fixed_seed_1000.zip --seed 2679 --episodes 3
        
        # 여러 시드 비교
        python record_driving_gif.py --model models/sac_fixed_seed_1000.zip --seeds 1000 2679 4657 --compare
        
        # 성공 vs 실패 비교
        python record_driving_gif.py --model models/sac_fixed_seed_1000.zip --success 1000 --failure 2679
    """
    import argparse
    
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
                       help="출력 파일명")
    parser.add_argument("--max-steps", type=int, default=1000,
                       help="최대 스텝 수")
    parser.add_argument("--compare", action="store_true",
                       help="여러 시드 비교 모드")
    parser.add_argument("--success", type=int, default=None,
                       help="성공 시드 (--failure와 함께 사용)")
    parser.add_argument("--failure", type=int, default=None,
                       help="실패 시드 (--success와 함께 사용)")
    
    args = parser.parse_args()
    
    # 성공 vs 실패 비교
    if args.success is not None and args.failure is not None:
        record_success_and_failure(
            model_path=args.model,
            success_seed=args.success,
            failure_seed=args.failure
        )
    
    # 여러 시드 비교
    elif args.compare and args.seeds:
        record_comparison(
            model_path=args.model,
            seeds=args.seeds
        )
    
    # 여러 에피소드
    elif args.seed and args.episodes > 1:
        record_multiple_episodes(
            model_path=args.model,
            seed=args.seed,
            n_episodes=args.episodes,
            output_dir="gifs"
        )
    
    # 단일 에피소드
    elif args.seed:
        record_driving_gif(
            model_path=args.model,
            seed=args.seed,
            output_name=args.output,
            max_steps=args.max_steps
        )
    
    # 시드 리스트 개별 녹화
    elif args.seeds:
        for seed in args.seeds:
            output_name = f"seed_{seed}.gif"
            record_driving_gif(
                model_path=args.model,
                seed=seed,
                output_name=output_name,
                max_steps=args.max_steps
            )
    
    else:
        print("❌ --seed 또는 --seeds 옵션이 필요합니다")
        print("사용법: python record_driving_gif.py --model MODEL_PATH --seed SEED")