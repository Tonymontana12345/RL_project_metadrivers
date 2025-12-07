"""
차량 궤적 시각화 평가 스크립트

실패한 에피소드에서 차량이 어디를 지나갔는지 궤적을 표시합니다.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from metadrive import MetaDriveEnv
from metadrive.utils.draw_top_down_map import draw_top_down_map
from stable_baselines3 import PPO, SAC, TD3
import os


def detect_algorithm(model_path):
    """모델 파일명에서 알고리즘 자동 감지"""
    model_name = os.path.basename(model_path).lower()
    if 'sac' in model_name:
        return 'sac'
    elif 'td3' in model_name:
        return 'td3'
    elif 'ppo' in model_name:
        return 'ppo'
    else:
        return 'ppo'


def load_model(model_path):
    """알고리즘에 맞는 모델 로드"""
    algorithm = detect_algorithm(model_path)
    
    if algorithm == 'ppo':
        return PPO.load(model_path)
    elif algorithm == 'sac':
        return SAC.load(model_path)
    elif algorithm == 'td3':
        return TD3.load(model_path)


def evaluate_with_trajectory(model_path, seed, n_episodes=5, save_dir="results/trajectories"):
    """
    모델을 평가하고 각 에피소드의 궤적을 저장
    
    Args:
        model_path: 모델 파일 경로
        seed: 테스트할 시드
        n_episodes: 에피소드 수
        save_dir: 저장 디렉토리
    """
    from config import FIXED_SEED_ENV_CONFIG
    
    print("\n" + "="*60)
    print(f"🚗 차량 궤적 평가 - Seed {seed}")
    print("="*60)
    print(f"모델: {model_path}")
    print(f"에피소드: {n_episodes}개")
    print("="*60 + "\n")
    
    # 디렉토리 생성
    os.makedirs(save_dir, exist_ok=True)
    
    # 모델 로드
    model = load_model(model_path)
    
    # 환경 생성 (config 사용!)
    env_config = FIXED_SEED_ENV_CONFIG.copy()
    env_config["start_seed"] = seed
    env_config["use_render"] = False
    
    env = MetaDriveEnv(env_config)
    
    # 맵 이미지 생성 (한 번만)
    env.reset(seed=seed)
    map_image = draw_top_down_map(env.current_map)
    
    # 각 에피소드 실행
    for episode in range(n_episodes):
        print(f"\n📍 에피소드 {episode + 1}/{n_episodes}")
        
        obs, info = env.reset(seed=seed)
        done = False
        total_reward = 0
        steps = 0
        
        # 궤적 저장
        trajectory = []
        
        while not done and steps < 2000:
            # 에이전트 위치 저장
            if hasattr(env.agent, 'position'):
                pos = env.agent.position
                trajectory.append([pos[0], pos[1]])
            
            # 액션 실행
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            steps += 1
        
        # 결과 출력
        success = info.get('arrive_dest', False)
        crash = info.get('crash', False)
        out_of_road = info.get('out_of_road', False)
        
        print(f"   보상: {total_reward:.2f}")
        print(f"   스텝: {steps}")
        print(f"   성공: {success}")
        print(f"   충돌: {crash}")
        print(f"   도로 이탈: {out_of_road}")
        
        # 궤적 시각화
        visualize_trajectory(
            map_image, 
            trajectory, 
            seed, 
            episode,
            total_reward,
            steps,
            success,
            crash,
            out_of_road,
            save_dir
        )
    
    env.close()
    
    print("\n" + "="*60)
    print(f"✅ 완료! 저장 위치: {save_dir}/")
    print("="*60 + "\n")


def visualize_trajectory(map_image, trajectory, seed, episode, reward, steps, 
                        success, crash, out_of_road, save_dir):
    """
    맵 위에 차량 궤적을 표시
    
    Args:
        map_image: 맵 이미지
        trajectory: 차량 궤적 [(x, y), ...]
        seed: 시드
        episode: 에피소드 번호
        reward: 총 보상
        steps: 스텝 수
        success: 성공 여부
        crash: 충돌 여부
        out_of_road: 도로 이탈 여부
        save_dir: 저장 디렉토리
    """
    fig, ax = plt.subplots(figsize=(14, 14))
    
    # 맵 표시
    ax.imshow(map_image, cmap="bone", alpha=0.8)
    
    if len(trajectory) > 0:
        trajectory = np.array(trajectory)
        
        # 맵 좌표계로 변환 (MetaDrive 좌표 -> 이미지 좌표)
        # MetaDrive는 중심 (0, 0), 이미지는 좌상단 (0, 0)
        img_height, img_width = map_image.shape[:2]
        
        # # 좌표 스케일링 (대략적인 변환)
        # scale =4# 조정 필요할 수 있음
        # center_x = img_width / 2
        # center_y = img_height / 2
        
        # traj_x = center_x + trajectory[:, 0] * scale
        # traj_y = center_y - trajectory[:, 1] * scale  # y축 반전

        min_x = np.min(trajectory[:, 0])
        max_x = np.max(trajectory[:, 0])
        min_y = np.min(trajectory[:, 1])
        max_y = np.max(trajectory[:, 1])

        # normalize world → pixel
        traj_x = (trajectory[:, 0] - min_x) / (max_x - min_x + 1e-6) * img_width
        traj_y = (trajectory[:, 1] - min_y) / (max_y - min_y + 1e-6) * img_height

        # flip y for image coordinate system
        traj_y = img_height - traj_y

      

        
        # 궤적 그리기 (시간에 따라 색상 변화)
        n_points = len(traj_x)
        
        # 1. 전체 경로 (얇은 선)
        ax.plot(traj_x, traj_y, 'cyan', linewidth=1, alpha=0.3, label='Path')
        
        # 2. 궤적을 색상으로 표시 (시작: 초록 -> 끝: 빨강/파랑)
        for i in range(n_points - 1):
            # 색상 계산 (시작: 초록, 중간: 노랑, 끝: 빨강 or 파랑)
            progress = i / max(n_points - 1, 1)
            
            if success:
                # 성공: 초록 -> 파랑
                color = (0, 1 - progress * 0.5, progress)
            elif crash or out_of_road:
                # 실패: 초록 -> 빨강
                color = (progress, 1 - progress, 0)
            else:
                # 기타: 초록 -> 노랑
                color = (progress, 1, 0)
            
            ax.plot(traj_x[i:i+2], traj_y[i:i+2], 
                   color=color, linewidth=3, alpha=0.7)
        
        # 3. 시작점 표시
        ax.scatter(traj_x[0], traj_y[0], 
                  c='green', s=300, marker='o', 
                  edgecolors='white', linewidths=2,
                  label='Start', zorder=10)
        
        # 4. 끝점 표시
        if success:
            end_color = 'blue'
            end_marker = '*'
            end_label = 'Goal Reached!'
        elif crash:
            end_color = 'red'
            end_marker = 'X'
            end_label = 'Crash'
        elif out_of_road:
            end_color = 'orange'
            end_marker = 'X'
            end_label = 'Out of Road'
        else:
            end_color = 'gray'
            end_marker = 'o'
            end_label = 'Timeout'
        
        ax.scatter(traj_x[-1], traj_y[-1], 
                  c=end_color, s=500, marker=end_marker, 
                  edgecolors='white', linewidths=3,
                  label=end_label, zorder=10)
        
        # 5. 중간 포인트 표시 (매 50스텝마다)
        step_interval = max(len(traj_x) // 10, 1)
        for i in range(0, len(traj_x), step_interval):
            ax.scatter(traj_x[i], traj_y[i], 
                      c='yellow', s=50, marker='o',
                      edgecolors='black', linewidths=1,
                      alpha=0.6, zorder=5)
    
    # 제목 및 정보
    status_emoji = '✅' if success else '❌'
    title = f"{status_emoji} Seed {seed} - Episode {episode + 1}\n"
    title += f"Reward: {reward:.1f} | Steps: {steps}"
    
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.legend(loc='upper right', fontsize=12, framealpha=0.9)
    ax.set_xticks([])
    ax.set_yticks([])
    
    # 통계 박스
    stats_text = f"Success: {success}\nCrash: {crash}\nOut of Road: {out_of_road}"
    ax.text(0.02, 0.02, stats_text, transform=ax.transAxes,
           fontsize=11, verticalalignment='bottom',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    # 저장
    filename = f"seed_{seed}_ep{episode + 1}_{'success' if success else 'fail'}.png"
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"   💾 저장: {filename}")


def compare_trajectories_grid(model_path, seeds, n_episodes_per_seed=3, 
                              save_path="results/trajectories_comparison.png"):
    """
    여러 시드의 궤적을 그리드로 비교
    
    Args:
        model_path: 모델 경로
        seeds: 시드 리스트
        n_episodes_per_seed: 시드당 에피소드 수
        save_path: 저장 경로
    """
    from config import FIXED_SEED_ENV_CONFIG
    
    print("\n" + "="*60)
    print("🗺️  궤적 비교 그리드 생성")
    print("="*60 + "\n")
    
    model = load_model(model_path)
    
    n_seeds = len(seeds)
    fig, axes = plt.subplots(n_seeds, n_episodes_per_seed, 
                            figsize=(8*n_episodes_per_seed, 8*n_seeds))
    
    if n_seeds == 1:
        axes = axes.reshape(1, -1)
    
    for i, seed in enumerate(seeds):
        print(f"처리 중: Seed {seed}")
        
        # 환경 생성 (config 사용!)
        env_config = FIXED_SEED_ENV_CONFIG.copy()
        env_config["start_seed"] = seed
        env_config["num_scenarios"] = 1
        env_config["use_render"] = False
        
        env = MetaDriveEnv(env_config)
        
        env.reset(seed=seed)
        map_image = draw_top_down_map(env.current_map)
        
        for ep in range(n_episodes_per_seed):
            obs, info = env.reset(seed=seed)
            done = False
            trajectory = []
            
            while not done and len(trajectory) < 2000:
                if hasattr(env.agent, 'position'):
                    pos = env.agent.position
                    trajectory.append([pos[0], pos[1]])
                
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
            
            # 해당 subplot에 그리기
            ax = axes[i, ep]
            ax.imshow(map_image, cmap="bone", alpha=0.8)
            
            if len(trajectory) > 0:
                trajectory = np.array(trajectory)
                img_height, img_width = map_image.shape[:2]
                scale = 4.0
                center_x = img_width / 2
                center_y = img_height / 2
                
                traj_x = center_x + trajectory[:, 0] * scale
                traj_y = center_y - trajectory[:, 1] * scale
                
                ax.plot(traj_x, traj_y, 'cyan', linewidth=2, alpha=0.6)
                ax.scatter(traj_x[0], traj_y[0], c='green', s=200, marker='o', zorder=10)
                
                success = info.get('arrive_dest', False)
                end_color = 'blue' if success else 'red'
                ax.scatter(traj_x[-1], traj_y[-1], c=end_color, s=200, marker='*', zorder=10)
            
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f'Seed {seed} - Ep {ep+1}', fontsize=11)
        
        env.close()
    
    plt.suptitle('Trajectory Comparison Across Seeds', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n💾 저장: {save_path}\n")


if __name__ == "__main__":
    """
    사용법:
        # 특정 시드에서 궤적 평가
        python visualize_trajectory.py --model models/sac_fixed_seed_1000.zip --seed 2679
        
        # 여러 에피소드
        python visualize_trajectory.py --model models/sac_fixed_seed_1000.zip --seed 2679 --episodes 10
        
        # 여러 시드 비교
        python visualize_trajectory.py --model models/sac_fixed_seed_1000.zip --seeds 2679 3286 4657 --compare
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="차량 궤적 시각화")
    parser.add_argument("--model", type=str, required=True,
                       help="모델 파일 경로")
    parser.add_argument("--seed", type=int, default=None,
                       help="단일 시드")
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                       help="여러 시드 (비교용)")
    parser.add_argument("--episodes", type=int, default=5,
                       help="에피소드 수")
    parser.add_argument("--compare", action="store_true",
                       help="비교 그리드 생성")
    parser.add_argument("--output", type=str, default="results/trajectories",
                       help="저장 디렉토리")
    
    args = parser.parse_args()
    
    if args.compare and args.seeds:
        # 비교 모드
        compare_trajectories_grid(
            args.model, 
            args.seeds, 
            n_episodes_per_seed=3,
            save_path=f"{args.output}/comparison.png"
        )
    elif args.seed:
        # 단일 시드
        evaluate_with_trajectory(
            args.model,
            args.seed,
            n_episodes=args.episodes,
            save_dir=args.output
        )
    elif args.seeds:
        # 여러 시드 (개별 평가)
        for seed in args.seeds:
            evaluate_with_trajectory(
                args.model,
                seed,
                n_episodes=args.episodes,
                save_dir=f"{args.output}/seed_{seed}"
            )
    else:
        print("❌ --seed 또는 --seeds 옵션이 필요합니다")
        print("사용법: python visualize_trajectory.py --model MODEL_PATH --seed SEED")

#######################################################################################
