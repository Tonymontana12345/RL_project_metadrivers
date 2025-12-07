"""
시드별 실제 맵 시각화 스크립트

블록 다이어그램 대신 실제 도로 지도 형태로 시각화합니다.
"""
import json
import os
import matplotlib.pyplot as plt
import numpy as np
from metadrive import MetaDriveEnv
from metadrive.utils.draw_top_down_map import draw_top_down_map
from config import FIXED_SEED, TEST_SEEDS


def visualize_map_topdown(seed, ax=None, title=None):
    """
    특정 시드의 맵을 탑다운 뷰로 시각화
    
    Args:
        seed: 맵 생성 시드
        ax: matplotlib axis (None이면 새로 생성)
        title: 제목 (None이면 자동 생성)
    
    Returns:
        ax: matplotlib axis
    """
    # 환경 생성
    env = MetaDriveEnv({
        "start_seed": seed,
        "num_scenarios": 1,
        "map": 3,
        "use_render": False,
        "manual_control": False,
        "log_level": 50,  # 로그 억제
    })
    
    # 환경 리셋하여 맵 생성
    env.reset(seed=seed)
    
    # 탑다운 맵 그리기
    map_image = draw_top_down_map(env.current_map)

    # ================================
    #  맵 픽셀 사이즈 정보
    # ================================
    img_h, img_w = map_image.shape[:2]

    # 시각화
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))
    
    ax.imshow(map_image, cmap="bone")
    ax.set_xticks([])
    ax.set_yticks([])
    
    if title is None:
        title = f'Seed {seed}'
    ax.set_title(title, fontsize=14, fontweight='bold', pad=10)
    
    # 테두리 추가
    for spine in ax.spines.values():
        spine.set_edgecolor('gray')
        spine.set_linewidth(2)
    
    env.close()
    
    return ax


def compare_all_seeds_topdown(seeds=None, save_path=None, cols=2):
    """
    모든 시드의 맵을 탑다운 뷰로 비교
    
    Args:
        seeds: 비교할 시드 리스트 (None이면 기본 시드 사용)
        save_path: 저장 경로
        cols: 열 개수
    """
    if seeds is None:
        seeds = [FIXED_SEED] + TEST_SEEDS
    
    print("\n" + "="*60)
    print("🗺️  시드별 실제 맵 시각화")
    print("="*60)
    
    n_seeds = len(seeds)
    rows = (n_seeds + cols - 1) // cols
    
    # 그리드 생성
    fig, axes = plt.subplots(rows, cols, figsize=(8*cols, 8*rows))
    
    # axes를 1차원 배열로 변환
    if n_seeds == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]
    
    for i, seed in enumerate(seeds):
        print(f"\n🔍 시드 {seed} 맵 생성 중...")
        try:
            visualize_map_topdown(seed, axes[i], title=f'Seed {seed}')
            print(f"   ✅ 완료")
        except Exception as e:
            print(f"   ❌ 오류: {e}")
            axes[i].text(0.5, 0.5, f'Seed {seed}\n(오류 발생)', 
                        ha='center', va='center', transform=axes[i].transAxes,
                        fontsize=14)
            axes[i].set_xticks([])
            axes[i].set_yticks([])
    
    # 남은 subplot 숨기기
    for i in range(n_seeds, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n💾 맵 비교 저장: {save_path}")
    else:
        plt.show()
    
    print("\n" + "="*60)
    print("✅ 맵 시각화 완료!")
    print("="*60 + "\n")


def visualize_single_map_large(seed, save_path=None):
    """
    단일 시드의 맵을 큰 크기로 시각화
    
    Args:
        seed: 맵 생성 시드
        save_path: 저장 경로
    """
    print(f"\n🗺️  시드 {seed} 맵 생성 중...\n")
    
    fig, ax = plt.subplots(figsize=(12, 12))
    visualize_map_topdown(seed, ax, title=f'Map for Seed {seed}')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"💾 맵 저장: {save_path}\n")
    else:
        plt.show()
    
    plt.close()


def create_map_grid_with_stats(seeds=None, evaluation_results=None, save_path=None):
    """
    맵과 통계를 함께 표시
    
    Args:
        seeds: 시드 리스트
        evaluation_results: 평가 결과 딕셔너리
        save_path: 저장 경로
    """
    if seeds is None:
        seeds = [FIXED_SEED] + TEST_SEEDS
    
    print("\n" + "="*60)
    print("🗺️  맵 + 통계 시각화")
    print("="*60)
    
    n_seeds = len(seeds)
    
    # 그리드 생성 (2열: 맵 + 통계)
    fig = plt.figure(figsize=(18, 6*n_seeds))
    
    for i, seed in enumerate(seeds):
        print(f"\n처리 중: 시드 {seed}...")
        
        # 맵 표시
        ax_map = plt.subplot(n_seeds, 2, 2*i + 1)
        try:
            visualize_map_topdown(seed, ax_map, title=f'Seed {seed} - Map')
        except Exception as e:
            print(f"   ❌ 맵 생성 오류: {e}")
            ax_map.text(0.5, 0.5, f'Error', ha='center', va='center', 
                       transform=ax_map.transAxes)
        
        # 통계 표시
        ax_stats = plt.subplot(n_seeds, 2, 2*i + 2)
        ax_stats.axis('off')
        
        if evaluation_results and str(seed) in evaluation_results.get("seed_results", {}):
            stats = evaluation_results["seed_results"][str(seed)]
            
            stats_text = f"""
Seed {seed} - Performance Statistics

Mean Reward: {stats['mean_reward']:.2f} ± {stats['std_reward']:.2f}
Mean Episode Length: {stats['mean_length']:.1f} steps

Success Rate: {stats['success_rate']*100:.1f}%
Crash Rate: {stats['crash_rate']*100:.1f}%
Out of Road Rate: {stats['out_of_road_rate']*100:.1f}%

Min Reward: {stats['min_reward']:.2f}
Max Reward: {stats['max_reward']:.2f}
            """
            
            # 성공률에 따른 색상
            if stats['success_rate'] >= 0.7:
                bg_color = 'lightgreen'
            elif stats['success_rate'] >= 0.3:
                bg_color = 'lightyellow'
            else:
                bg_color = 'lightcoral'
            
            ax_stats.text(0.1, 0.5, stats_text.strip(), 
                         fontsize=12, 
                         verticalalignment='center',
                         family='monospace',
                         bbox=dict(boxstyle='round', facecolor=bg_color, alpha=0.8))
        else:
            ax_stats.text(0.5, 0.5, 'No evaluation data', 
                         ha='center', va='center',
                         transform=ax_stats.transAxes)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n💾 저장: {save_path}")
    else:
        plt.show()
    
    print("\n" + "="*60)
    print("✅ 완료!")
    print("="*60 + "\n")


if __name__ == "__main__":
    """
    메인 실행
    
    사용법:
        # 모든 시드 맵 비교
        python visualize_maps_topdown.py
        
        # 특정 시드만
        python visualize_maps_topdown.py --seeds 1000 2679
        
        # 단일 맵 크게 보기
        python visualize_maps_topdown.py --seed 1000 --large
        
        # 통계와 함께
        python visualize_maps_topdown.py --with-stats
    """
    import argparse
    import json
    import os
    from utils.path_utils import get_result_path
    
    parser = argparse.ArgumentParser(description="시드별 실제 맵 시각화")
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                       help="시각화할 시드 리스트")
    parser.add_argument("--seed", type=int, default=None,
                       help="단일 시드 (--large와 함께 사용)")
    parser.add_argument("--large", action="store_true",
                       help="큰 크기로 단일 맵 시각화")
    parser.add_argument("--with-stats", action="store_true",
                       help="평가 통계와 함께 표시")
    parser.add_argument("--save", action="store_true",
                       help="결과 저장")
    parser.add_argument("--cols", type=int, default=2,
                       help="열 개수")
    
    args = parser.parse_args()
    
    # 시드 설정
    if args.seed:
        seeds = [args.seed]
    elif args.seeds:
        seeds = args.seeds
    else:
        seeds = [FIXED_SEED] + TEST_SEEDS
    
    # 단일 맵 크게 보기
    if args.large:
        if len(seeds) != 1:
            print("⚠️  --large 옵션은 --seed와 함께 사용하세요")
            seeds = [seeds[0]]
        
        save_path = get_result_path(f"seed_{seeds[0]}_map_large.png") if args.save else None
        visualize_single_map_large(seeds[0], save_path)
    
    # 통계와 함께 표시
    elif args.with_stats:
        # 평가 결과 로드
        eval_path = get_result_path("evaluation_results.json")
        evaluation_results = None
        
        if os.path.exists(eval_path):
            with open(eval_path, 'r') as f:
                evaluation_results = json.load(f)
        else:
            print("⚠️  evaluation_results.json을 찾을 수 없습니다")
        
        save_path = get_result_path("seed_maps_with_stats.png") if args.save else None
        create_map_grid_with_stats(seeds, evaluation_results, save_path)
    
    # 기본: 맵 비교
    else:
        save_path = get_result_path("seed_maps_topdown.png") if args.save else None
        compare_all_seeds_topdown(seeds, save_path, cols=args.cols)