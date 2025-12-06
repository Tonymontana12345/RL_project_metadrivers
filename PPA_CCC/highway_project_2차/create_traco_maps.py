"""
TRACO 형태 맵 생성 스크립트
MetaDrive의 draw_top_down_map을 사용하여 맵과 궤적을 함께 시각화합니다.

TRACO = Trajectory + Map (궤적과 맵을 결합한 시각화)
"""

import sys
import os
sys.path.insert(0, '.')

import matplotlib.pyplot as plt
import numpy as np
from metadrive import MetaDriveEnv
from metadrive.utils.draw_top_down_map import draw_top_down_map
from stable_baselines3 import PPO
from config import FIXED_SEED_ENV_CONFIG
from utils.path_utils import get_result_path


def world_to_pixel_coords(trajectory, map_image_shape, map_bounds=None):
    """
    세계 좌표를 픽셀 좌표로 변환
    
    Args:
        trajectory: [(x, y), ...] 형태의 궤적
        map_image_shape: (height, width) 맵 이미지 크기
        map_bounds: (min_x, max_x, min_y, max_y) 맵 경계 (None이면 자동 계산)
    
    Returns:
        (traj_x, traj_y): 픽셀 좌표
    """
    if len(trajectory) == 0:
        return np.array([]), np.array([])
    
    trajectory = np.array(trajectory)
    img_height, img_width = map_image_shape[:2]
    
    if map_bounds is None:
        # 자동으로 경계 계산
        min_x = np.min(trajectory[:, 0])
        max_x = np.max(trajectory[:, 0])
        min_y = np.min(trajectory[:, 1])
        max_y = np.max(trajectory[:, 1])
        
        # 여유 공간 추가
        margin_x = (max_x - min_x) * 0.1 if max_x != min_x else 10
        margin_y = (max_y - min_y) * 0.1 if max_y != min_y else 10
        
        min_x -= margin_x
        max_x += margin_x
        min_y -= margin_y
        max_y += margin_y
    else:
        min_x, max_x, min_y, max_y = map_bounds
    
    # 정규화
    if max_x - min_x > 1e-6:
        traj_x = (trajectory[:, 0] - min_x) / (max_x - min_x) * img_width
    else:
        traj_x = np.ones(len(trajectory)) * img_width / 2
    
    if max_y - min_y > 1e-6:
        traj_y = (trajectory[:, 1] - min_y) / (max_y - min_y) * img_height
    else:
        traj_y = np.ones(len(trajectory)) * img_height / 2
    
    # y축 반전 (이미지 좌표계)
    traj_y = img_height - traj_y
    
    return traj_x, traj_y


def create_traco_map(seed=None, model_path=None, map_string=None, n_episodes=3, resolution=(1024, 1024), save_path=None):
    """
    TRACO 형태 맵 생성 (맵 + 궤적)
    
    Args:
        seed: 맵 시드 (map_string이 None일 때 사용)
        model_path: 모델 경로
        map_string: 블록 타입 ID 문자열 (예: "SSSS", "SCSCS", "XOXO") - 우선순위 높음
        n_episodes: 에피소드 수
        resolution: 맵 해상도
        save_path: 저장 경로
    
    Returns:
        fig: matplotlib figure
    """
    if map_string:
        print(f"\n🗺️  TRACO 맵 생성 중: Map '{map_string}'")
        map_display = f"Map '{map_string}'"
    else:
        print(f"\n🗺️  TRACO 맵 생성 중: Seed {seed}")
        map_display = f"Seed {seed}"
    print("="*60)
    
    # 모델 로드
    if model_path:
        print("📦 모델 로드 중...")
        model = PPO.load(model_path)
        print("✅ 모델 로드 완료")
    else:
        model = None
        print("⚠️  모델 없이 맵만 생성합니다 (궤적 없음)")
    
    # 환경 생성
    env_config = FIXED_SEED_ENV_CONFIG.copy()
    
    if map_string:
        # 블록 타입 ID 문자열로 맵 생성
        env_config["map"] = map_string
        # 맵 문자열 사용 시에도 시드를 설정하면 블록의 세부 구조가 달라짐
        if seed is not None:
            env_config["start_seed"] = seed
        else:
            env_config["start_seed"] = 0
    else:
        env_config["start_seed"] = seed
        env_config["map"] = env_config.get("map", 3)  # 기본값 유지
    
    env_config["num_scenarios"] = 1
    env_config["use_render"] = False
    
    env = MetaDriveEnv(env_config)
    env.reset(seed=seed if seed else 0)
    
    # MetaDrive의 draw_top_down_map 사용
    print("📐 맵 렌더링 중...")
    map_image = draw_top_down_map(env.current_map, resolution=resolution, semantic_map=True)
    print(f"✅ 맵 렌더링 완료: {map_image.shape}")
    
    # 각 에피소드의 궤적 수집
    all_trajectories = []
    all_results = []
    
    if model:
        for episode in range(n_episodes):
            print(f"\n📍 에피소드 {episode + 1}/{n_episodes} 실행 중...")
            
            obs, info = env.reset(seed=seed if seed else 0)
            done = False
            trajectory = []
            
            while not done and len(trajectory) < 2000:
                if hasattr(env.agent, 'position'):
                    pos = env.agent.position
                    trajectory.append([pos[0], pos[1]])
                
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
            
            all_trajectories.append(trajectory)
            
            result = {
                'success': info.get('arrive_dest', False),
                'crash': info.get('crash', False) or info.get('crash_vehicle', False),
                'out_of_road': info.get('out_of_road', False),
                'steps': len(trajectory)
            }
            all_results.append(result)
            
            status = "✅ 성공" if result['success'] else "❌ 실패"
            print(f"   {status} (스텝: {result['steps']})")
    else:
        print("\n⚠️  모델이 없어 궤적을 생성하지 않습니다.")
        all_results = [{'success': False, 'crash': False, 'out_of_road': False, 'steps': 0}] * n_episodes
    
    env.close()
    
    # 시각화
    print("\n🎨 시각화 생성 중...")
    fig, ax = plt.subplots(figsize=(12, 12))
    
    # 맵 표시
    ax.imshow(map_image, cmap='bone', alpha=0.9, origin='upper')
    
    # 모든 궤적의 경계 계산
    all_points = []
    for traj in all_trajectories:
        if len(traj) > 0:
            all_points.extend(traj)
    
    if len(all_points) > 0:
        all_points = np.array(all_points)
        min_x, max_x = np.min(all_points[:, 0]), np.max(all_points[:, 0])
        min_y, max_y = np.min(all_points[:, 1]), np.max(all_points[:, 1])
        
        # 여유 공간 추가
        margin_x = (max_x - min_x) * 0.15 if max_x != min_x else 20
        margin_y = (max_y - min_y) * 0.15 if max_y != min_y else 20
        
        map_bounds = (min_x - margin_x, max_x + margin_x, 
                     min_y - margin_y, max_y + margin_y)
    else:
        map_bounds = None
    
    # 각 에피소드의 궤적 그리기
    colors = ['#00ff00', '#00aaff', '#ffaa00', '#ff0000', '#aa00ff']
    
    for ep_idx, (trajectory, result) in enumerate(zip(all_trajectories, all_results)):
        if len(trajectory) == 0:
            continue
        
        traj_x, traj_y = world_to_pixel_coords(trajectory, map_image.shape, map_bounds)
        
        if len(traj_x) == 0:
            continue
        
        color = colors[ep_idx % len(colors)]
        
        # 궤적 그리기 (시간에 따라 색상 변화)
        n_points = len(traj_x)
        for i in range(n_points - 1):
            progress = i / max(n_points - 1, 1)
            
            if result['success']:
                # 성공: 초록 -> 파랑
                alpha = 0.6 + 0.3 * progress
                line_color = (0, 1 - progress * 0.3, progress)
            elif result['crash']:
                # 충돌: 초록 -> 빨강
                alpha = 0.6 + 0.3 * progress
                line_color = (progress, 1 - progress, 0)
            else:
                # 기타: 초록 -> 노랑
                alpha = 0.5 + 0.3 * progress
                line_color = (progress, 1, 0)
            
            ax.plot(traj_x[i:i+2], traj_y[i:i+2], 
                   color=line_color, linewidth=2.5, alpha=alpha, zorder=5)
        
        # 시작점 (초록 원)
        ax.scatter(traj_x[0], traj_y[0], 
                  c='green', s=400, marker='o', 
                  edgecolors='white', linewidths=3,
                  label=f'Start (Ep {ep_idx+1})' if ep_idx == 0 else '',
                  zorder=10, alpha=0.9)
        
        # 끝점
        if result['success']:
            end_color = '#0066ff'
            end_marker = '*'
            end_size = 600
            end_label = f'Goal (Ep {ep_idx+1})' if ep_idx == 0 else ''
        elif result['crash']:
            end_color = '#ff0000'
            end_marker = 'X'
            end_size = 500
            end_label = f'Crash (Ep {ep_idx+1})' if ep_idx == 0 else ''
        elif result['out_of_road']:
            end_color = '#ff8800'
            end_marker = 'X'
            end_size = 500
            end_label = f'Out of Road (Ep {ep_idx+1})' if ep_idx == 0 else ''
        else:
            end_color = '#888888'
            end_marker = 'o'
            end_size = 400
            end_label = f'Timeout (Ep {ep_idx+1})' if ep_idx == 0 else ''
        
        ax.scatter(traj_x[-1], traj_y[-1], 
                  c=end_color, s=end_size, marker=end_marker, 
                  edgecolors='white', linewidths=3,
                  label=end_label,
                  zorder=10, alpha=0.9)
    
    # 제목 및 스타일
    success_count = sum(1 for r in all_results if r['success'])
    if map_string:
        title = f'TRACO Map - "{map_string}"\n'
        if model:
            title += f'Success: {success_count}/{n_episodes} episodes'
        else:
            title += 'Map Only (No Trajectory)'
    else:
        title = f'TRACO Map - Seed {seed}\n'
        title += f'Success: {success_count}/{n_episodes} episodes'
    
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(loc='upper right', fontsize=11, framealpha=0.9, 
             facecolor='white', edgecolor='gray')
    
    # 통계 정보 박스
    if map_string:
        stats_text = f"Map: {map_string}\n"
        block_names = {
            'S': 'Straight', 'r': 'InRamp', 'O': 'Roundabout', 'y': 'Merge',
            '$': 'Tollgate', 'T': 'T-intersection', 'C': 'Circular',
            'R': 'OutRamp', 'X': 'Intersection', 'Y': 'Split', 'P': 'Parking'
        }
        blocks_desc = ' → '.join([block_names.get(b, b) for b in map_string])
        stats_text += f"Blocks: {blocks_desc}\n"
    else:
        stats_text = f"Seed: {seed}\n"
    
    if model:
        stats_text += f"Episodes: {n_episodes}\n"
        stats_text += f"Success Rate: {success_count}/{n_episodes} ({success_count/n_episodes*100:.0f}%)\n"
        
        if all_results:
            avg_steps = np.mean([r['steps'] for r in all_results])
            stats_text += f"Avg Steps: {avg_steps:.0f}"
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
           fontsize=11, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
        print(f"\n💾 저장 완료: {save_path}")
    else:
        plt.show()
    
    print("="*60)
    print("✅ TRACO 맵 생성 완료!")
    print("="*60 + "\n")
    
    return fig


def create_traco_grid(seeds, model_path, n_episodes=3, resolution=(1024, 1024), save_path=None):
    """
    여러 시드의 TRACO 맵을 그리드로 생성
    
    Args:
        seeds: 시드 리스트
        model_path: 모델 경로
        n_episodes: 시드당 에피소드 수
        resolution: 맵 해상도
        save_path: 저장 경로
    """
    print("\n" + "="*60)
    print("🗺️  TRACO 맵 그리드 생성")
    print("="*60)
    print(f"시드: {seeds}")
    print(f"에피소드/시드: {n_episodes}")
    print("="*60 + "\n")
    
    n_seeds = len(seeds)
    cols = 2
    rows = (n_seeds + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(12*cols, 12*rows))
    
    if n_seeds == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if isinstance(axes, np.ndarray) else list(axes)
    
    # 모델 로드 (한 번만)
    model = PPO.load(model_path)
    
    for idx, seed in enumerate(seeds):
        print(f"\n[{idx+1}/{n_seeds}] Seed {seed} 처리 중...")
        
        ax = axes[idx]
        
        # 환경 생성
        env_config = FIXED_SEED_ENV_CONFIG.copy()
        env_config["start_seed"] = seed
        env_config["num_scenarios"] = 1
        env_config["use_render"] = False
        
        env = MetaDriveEnv(env_config)
        env.reset(seed=seed)
        
        # 맵 렌더링
        map_image = draw_top_down_map(env.current_map, resolution=resolution, semantic_map=True)
        ax.imshow(map_image, cmap='bone', alpha=0.9, origin='upper')
        
        # 궤적 수집 및 그리기
        all_points = []
        all_trajectories = []
        all_results = []
        
        for episode in range(n_episodes):
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
            
            all_trajectories.append(trajectory)
            if len(trajectory) > 0:
                all_points.extend(trajectory)
            
            result = {
                'success': info.get('arrive_dest', False),
                'crash': info.get('crash', False) or info.get('crash_vehicle', False),
                'out_of_road': info.get('out_of_road', False),
                'steps': len(trajectory)
            }
            all_results.append(result)
        
        env.close()
        
        # 경계 계산
        if len(all_points) > 0:
            all_points = np.array(all_points)
            min_x, max_x = np.min(all_points[:, 0]), np.max(all_points[:, 0])
            min_y, max_y = np.min(all_points[:, 1]), np.max(all_points[:, 1])
            margin_x = (max_x - min_x) * 0.15 if max_x != min_x else 20
            margin_y = (max_y - min_y) * 0.15 if max_y != min_y else 20
            map_bounds = (min_x - margin_x, max_x + margin_x, 
                         min_y - margin_y, max_y + margin_y)
        else:
            map_bounds = None
        
        # 궤적 그리기
        colors = ['#00ff00', '#00aaff', '#ffaa00', '#ff0000', '#aa00ff']
        
        for ep_idx, (trajectory, result) in enumerate(zip(all_trajectories, all_results)):
            if len(trajectory) == 0:
                continue
            
            traj_x, traj_y = world_to_pixel_coords(trajectory, map_image.shape, map_bounds)
            
            if len(traj_x) == 0:
                continue
            
            # 궤적 선 그리기
            n_points = len(traj_x)
            for i in range(n_points - 1):
                progress = i / max(n_points - 1, 1)
                
                if result['success']:
                    line_color = (0, 1 - progress * 0.3, progress)
                elif result['crash']:
                    line_color = (progress, 1 - progress, 0)
                else:
                    line_color = (progress, 1, 0)
                
                ax.plot(traj_x[i:i+2], traj_y[i:i+2], 
                       color=line_color, linewidth=2, alpha=0.7, zorder=5)
            
            # 시작점
            ax.scatter(traj_x[0], traj_y[0], 
                      c='green', s=300, marker='o', 
                      edgecolors='white', linewidths=2, zorder=10)
            
            # 끝점
            if result['success']:
                end_color = '#0066ff'
                end_marker = '*'
            elif result['crash']:
                end_color = '#ff0000'
                end_marker = 'X'
            elif result['out_of_road']:
                end_color = '#ff8800'
                end_marker = 'X'
            else:
                end_color = '#888888'
                end_marker = 'o'
            
            ax.scatter(traj_x[-1], traj_y[-1], 
                      c=end_color, s=400, marker=end_marker, 
                      edgecolors='white', linewidths=2, zorder=10)
        
        # 제목
        success_count = sum(1 for r in all_results if r['success'])
        ax.set_title(f'Seed {seed}\nSuccess: {success_count}/{n_episodes}', 
                    fontsize=14, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])
    
    # 남은 subplot 숨기기
    for i in range(n_seeds, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle('TRACO Maps - Trajectory + Map Visualization', 
                fontsize=18, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"\n💾 저장 완료: {save_path}")
    else:
        plt.show()
    
    plt.close()
    
    print("\n" + "="*60)
    print("✅ TRACO 그리드 생성 완료!")
    print("="*60 + "\n")


if __name__ == "__main__":
    """
    사용법:
        # 단일 시드 TRACO 맵
        python create_traco_maps.py --seed 8576 --model models/ppo_balanced_6seeds_v4_800k.zip
        
        # 여러 시드 그리드
        python create_traco_maps.py --seeds 8576 6339 1409 4806 --model models/ppo_balanced_6seeds_v4_800k.zip --grid
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="TRACO 형태 맵 생성")
    parser.add_argument("--seed", type=int, default=None,
                       help="단일 시드")
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                       help="여러 시드 (그리드용)")
    parser.add_argument("--map", type=str, default=None,
                       help="블록 타입 ID 문자열 (예: SSSS, SCSCS, XOXO)")
    parser.add_argument("--maps", type=str, nargs="+", default=None,
                       help="여러 맵 문자열 (그리드용)")
    parser.add_argument("--model", type=str, default=None,
                       help="모델 경로 (없으면 맵만 생성)")
    parser.add_argument("--episodes", type=int, default=3,
                       help="에피소드 수")
    parser.add_argument("--resolution", type=int, nargs=2, default=[1024, 1024],
                       help="맵 해상도 [width, height]")
    parser.add_argument("--grid", action="store_true",
                       help="그리드 형태로 생성")
    parser.add_argument("--save", action="store_true",
                       help="결과 저장")
    
    args = parser.parse_args()
    
    # 기본 모델 경로
    if args.model is None and (args.seed or args.seeds):
        args.model = "models/ppo_balanced_6seeds_v4_800k.zip"
    
    if args.map:
        # 블록 타입 ID 문자열로 맵 생성
        save_path = get_result_path(f"traco_map_{args.map}.png") if args.save else None
        create_traco_map(seed=None, model_path=args.model, map_string=args.map, 
                        n_episodes=args.episodes, resolution=tuple(args.resolution), 
                        save_path=save_path)
    elif args.maps:
        # 여러 맵 문자열 그리드
        print("⚠️  여러 맵 문자열 그리드는 아직 구현되지 않았습니다.")
        print("   각 맵을 개별적으로 생성하세요:")
        for map_str in args.maps:
            save_path = get_result_path(f"traco_map_{map_str}.png") if args.save else None
            create_traco_map(seed=None, model_path=args.model, map_string=map_str,
                           n_episodes=args.episodes, resolution=tuple(args.resolution),
                           save_path=save_path)
    elif args.seed:
        # 단일 시드
        save_path = get_result_path(f"traco_seed_{args.seed}.png") if args.save else None
        create_traco_map(seed=args.seed, model_path=args.model, map_string=None,
                        n_episodes=args.episodes, resolution=tuple(args.resolution), 
                        save_path=save_path)
    elif args.seeds:
        # 여러 시드
        save_path = get_result_path("traco_maps_grid.png") if args.save else None
        create_traco_grid(args.seeds, args.model, args.episodes, 
                         tuple(args.resolution), save_path)
    else:
        print("❌ --seed, --seeds, --map 또는 --maps 옵션이 필요합니다")
        print("\n사용법:")
        print("  # 블록 타입 ID 문자열로 맵 생성 (맵만)")
        print("  python create_traco_maps.py --map SSSS --save")
        print("\n  # 블록 타입 ID 문자열로 맵 생성 (맵 + 궤적)")
        print("  python create_traco_maps.py --map SCSCS --model models/ppo_balanced_6seeds_v4_800k.zip --save")
        print("\n  # 시드로 맵 생성")
        print("  python create_traco_maps.py --seed 8576 --model models/ppo_balanced_6seeds_v4_800k.zip --save")

