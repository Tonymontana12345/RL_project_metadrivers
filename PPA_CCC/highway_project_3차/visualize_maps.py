"""
시드별 맵 시각화 스크립트

각 시드마다 생성되는 맵의 구조를 시각화하고 비교합니다.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from metadrive import MetaDriveEnv
from config import FIXED_SEED, TEST_SEEDS

def visualize_seed_map(seed, ax=None, show_info=True):
    """
    특정 시드의 맵을 시각화
    
    Args:
        seed: 맵 생성 시드
        ax: matplotlib axis (None이면 새로 생성)
        show_info: 맵 정보 표시 여부
    
    Returns:
        ax: matplotlib axis
        map_info: 맵 정보 딕셔너리
    """
    # 환경 생성 (렌더링 없이)
    env = MetaDriveEnv({
        "start_seed": seed,
        "num_scenarios": 1,
        "map": 5,  # 5개 블록
        "use_render": False,
        "manual_control": False,
    })
    
    # 환경 리셋하여 맵 생성
    env.reset(seed=seed)
    
    # 맵 정보 추출
    map_info = {
        "seed": seed,
        "blocks": [],
        "total_length": 0,
    }
    
    # 현재 맵의 블록 정보 가져오기
    current_map = env.current_map
    if hasattr(current_map, 'blocks'):
        for block in current_map.blocks:
            block_type = type(block).__name__
            map_info["blocks"].append(block_type)
    
    # 맵 길이
    if hasattr(env.current_map, 'road_network'):
        map_info["total_length"] = len(env.current_map.road_network.graph)
    
    # 시각화
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    
    # 맵 렌더링 (탑뷰)
    try:
        # 에이전트 위치 가져오기
        agent = env.agent
        if agent and hasattr(agent, 'position'):
            agent_pos = agent.position
            
            # 도로 네트워크 그리기
            if hasattr(env.current_map, 'road_network'):
                road_network = env.current_map.road_network
                
                # 모든 도로 그리기
                for road_id, road in road_network.graph.items():
                    if hasattr(road, 'lanes'):
                        for lane in road.lanes:
                            if hasattr(lane, 'line_types'):
                                # 차선 그리기 (간단한 버전)
                                ax.plot([0, 100], [0, 0], 'gray', alpha=0.3, linewidth=1)
            
            # 시작점 표시
            ax.plot(0, 0, 'go', markersize=15, label='Start')
            
            # 목적지 표시 (대략적 위치)
            ax.plot(100, 0, 'r*', markersize=20, label='Goal')
    
    except Exception as e:
        # 렌더링 실패 시 기본 정보만 표시
        ax.text(0.5, 0.5, f'Seed {seed}\n(렌더링 불가)', 
                ha='center', va='center', fontsize=14, transform=ax.transAxes)
    
    # 맵 정보 표시
    if show_info:
        info_text = f"Seed: {seed}\n"
        if map_info["blocks"]:
            info_text += f"Blocks: {len(map_info['blocks'])}\n"
            info_text += f"Types: {', '.join(map_info['blocks'][:3])}"
            if len(map_info['blocks']) > 3:
                info_text += "..."
        
        ax.text(0.02, 0.98, info_text, 
                transform=ax.transAxes, 
                fontsize=10, 
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    ax.set_title(f'Map for Seed {seed}', fontsize=12, fontweight='bold')
    ax.set_xlabel('X Position (m)')
    ax.set_ylabel('Y Position (m)')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')
    ax.axis('equal')
    
    # 환경 종료
    env.close()
    
    return ax, map_info


def visualize_seed_map_simple(seed, ax=None):
    """
    시드별 맵을 간단하게 시각화 (블록 정보만)
    
    Args:
        seed: 맵 생성 시드
        ax: matplotlib axis
    
    Returns:
        ax: matplotlib axis
        map_info: 맵 정보
    """
    # 환경 생성
    env = MetaDriveEnv({
        "start_seed": seed,
        "num_scenarios": 1,
        "map": 5,
        "use_render": False,
    })
    
    env.reset(seed=seed)
    
    # 맵 정보 추출
    map_info = {
        "seed": seed,
        "blocks": [],
        "block_types": [],
    }
    
    # 블록 정보 가져오기
    current_map = env.current_map
    if hasattr(current_map, 'blocks'):
        for i, block in enumerate(current_map.blocks):
            block_type = type(block).__name__
            map_info["blocks"].append(block_type)
            
            # 블록 타입 약어 (대소문자 구분 없이)
            block_type_lower = block_type.lower()
            
            if "first" in block_type_lower:
                map_info["block_types"].append("1")  # 시작 블록
            elif "straight" in block_type_lower:
                map_info["block_types"].append("S")
            elif "curve" in block_type_lower or "circular" in block_type_lower:
                map_info["block_types"].append("C")
            elif "inramp" in block_type_lower:
                map_info["block_types"].append("r")
            elif "outramp" in block_type_lower:
                map_info["block_types"].append("R")
            elif "roundabout" in block_type_lower:
                map_info["block_types"].append("O")
            elif "tintersection" in block_type_lower:
                map_info["block_types"].append("T")
            elif "intersection" in block_type_lower:
                map_info["block_types"].append("X")
            elif "fork" in block_type_lower:
                map_info["block_types"].append("F")
            elif "merge" in block_type_lower:
                map_info["block_types"].append("M")
            else:
                # 알 수 없는 타입은 이름 출력하여 디버깅
                map_info["block_types"].append(f"?({block_type[:3]})")
    
    # 시각화
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 3))
    
    # 블록을 순서대로 그리기
    colors = {
        "1": "gold",          # 시작 블록
        "S": "lightblue",
        "C": "lightgreen",
        "r": "lightyellow",
        "R": "lightyellow",
        "O": "lightcoral",
        "X": "lightpink",
        "T": "lightpink",
        "F": "lavender",
        "M": "lavender",
    }
    
    # 기본 색상 (알 수 없는 타입용)
    default_color = "lightgray"
    
    block_width = 1.0
    for i, (block_name, block_type) in enumerate(zip(map_info["blocks"], map_info["block_types"])):
        # ? 타입인 경우 괄호 제거하여 색상 매칭
        block_type_clean = block_type.split('(')[0] if '(' in block_type else block_type
        
        # 블록 사각형 그리기
        rect = patches.Rectangle(
            (i * block_width, 0), 
            block_width, 
            1.0,
            linewidth=2,
            edgecolor='black',
            facecolor=colors.get(block_type_clean, default_color),
            alpha=0.7
        )
        ax.add_patch(rect)
        
        # 블록 타입 텍스트
        ax.text(i * block_width + block_width/2, 0.5, 
                block_type, 
                ha='center', va='center', 
                fontsize=16, fontweight='bold')
        
        # 블록 이름 (아래)
        short_name = block_name.replace("Block", "").replace("Straight", "Str").replace("Circular", "Cir")
        ax.text(i * block_width + block_width/2, -0.3, 
                short_name, 
                ha='center', va='top', 
                fontsize=8, rotation=0)
    
    # 축 설정
    ax.set_xlim(-0.5, len(map_info["blocks"]) * block_width + 0.5)
    ax.set_ylim(-0.5, 1.5)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # 제목
    block_sequence = ''.join(map_info["block_types"])
    ax.set_title(f'Seed {seed}: {block_sequence}', 
                 fontsize=12, fontweight='bold', pad=20)
    
    env.close()
    
    return ax, map_info


def compare_all_seeds(seeds=None, save_path=None):
    """
    모든 시드의 맵을 비교
    
    Args:
        seeds: 비교할 시드 리스트 (None이면 기본 시드 사용)
        save_path: 저장 경로
    """
    if seeds is None:
        seeds = [FIXED_SEED] + TEST_SEEDS
    
    print("\n" + "="*60)
    print("🗺️  시드별 맵 비교")
    print("="*60)
    
    # 그리드 생성
    n_seeds = len(seeds)
    fig, axes = plt.subplots(n_seeds, 1, figsize=(12, 3 * n_seeds))
    
    if n_seeds == 1:
        axes = [axes]
    
    map_infos = []
    
    for i, seed in enumerate(seeds):
        print(f"\n🔍 시드 {seed} 분석 중...")
        try:
            ax, map_info = visualize_seed_map_simple(seed, axes[i])
            map_infos.append(map_info)
            print(f"   블록 구성: {''.join(map_info['block_types'])}")
        except Exception as e:
            print(f"   ❌ 오류: {e}")
            axes[i].text(0.5, 0.5, f'Seed {seed}\n(오류 발생)', 
                        ha='center', va='center', transform=axes[i].transAxes)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n💾 맵 비교 저장: {save_path}")
    else:
        plt.show()
    
    plt.close()
    
    # 요약 출력
    print("\n" + "="*60)
    print("📊 맵 구성 요약")
    print("="*60)
    
    for map_info in map_infos:
        block_seq = ''.join(map_info['block_types'])
        print(f"시드 {map_info['seed']:4d}: {block_seq:10s} ({len(map_info['blocks'])}개 블록)")
    
    print("="*60 + "\n")
    
    # 블록 타입 설명
    print("📖 블록 타입 설명:")
    print("  S: Straight (직선)")
    print("  C: Circular/Curve (커브)")
    print("  r: InRamp (진입로)")
    print("  R: OutRamp (출구)")
    print("  O: Roundabout (로터리)")
    print("  X: Intersection (교차로)")
    print("  T: TIntersection (T자 교차로)")
    print()


def analyze_seed_difficulty(seed):
    """
    시드별 난이도 분석
    
    Args:
        seed: 분석할 시드
    
    Returns:
        dict: 난이도 정보
    """
    env = MetaDriveEnv({
        "start_seed": seed,
        "num_scenarios": 1,
        "map": 5,
        "use_render": False,
    })
    
    env.reset(seed=seed)
    
    difficulty = {
        "seed": seed,
        "num_curves": 0,
        "num_intersections": 0,
        "num_straight": 0,
        "difficulty_score": 0,
    }
    
    # 블록 분석
    if hasattr(env.current_map, 'blocks'):
        for block in env.current_map.blocks:
            block_type = type(block).__name__
            
            if "Straight" in block_type:
                difficulty["num_straight"] += 1
            elif "Curve" in block_type or "Circular" in block_type:
                difficulty["num_curves"] += 1
                difficulty["difficulty_score"] += 2  # 커브는 어려움
            elif "Intersection" in block_type or "TIntersection" in block_type:
                difficulty["num_intersections"] += 1
                difficulty["difficulty_score"] += 3  # 교차로는 매우 어려움
            elif "Roundabout" in block_type:
                difficulty["difficulty_score"] += 4  # 로터리는 가장 어려움
    
    env.close()
    
    return difficulty


def compare_seed_difficulty(seeds=None, save_path=None):
    """
    시드별 난이도 비교
    
    Args:
        seeds: 비교할 시드 리스트
        save_path: 저장 경로
    """
    if seeds is None:
        seeds = [FIXED_SEED] + TEST_SEEDS
    
    print("\n" + "="*60)
    print("📊 시드별 난이도 분석")
    print("="*60)
    
    difficulties = []
    
    for seed in seeds:
        print(f"\n분석 중: 시드 {seed}...")
        diff = analyze_seed_difficulty(seed)
        difficulties.append(diff)
        
        print(f"  직선: {diff['num_straight']}개")
        print(f"  커브: {diff['num_curves']}개")
        print(f"  교차로: {diff['num_intersections']}개")
        print(f"  난이도 점수: {diff['difficulty_score']}")
    
    # 시각화
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # 1. 블록 타입 분포
    ax1 = axes[0]
    seeds_list = [d['seed'] for d in difficulties]
    straight = [d['num_straight'] for d in difficulties]
    curves = [d['num_curves'] for d in difficulties]
    intersections = [d['num_intersections'] for d in difficulties]
    
    x = np.arange(len(seeds_list))
    width = 0.25
    
    ax1.bar(x - width, straight, width, label='Straight', alpha=0.8)
    ax1.bar(x, curves, width, label='Curves', alpha=0.8)
    ax1.bar(x + width, intersections, width, label='Intersections', alpha=0.8)
    
    ax1.set_xlabel('Seed', fontsize=12)
    ax1.set_ylabel('Count', fontsize=12)
    ax1.set_title('Block Type Distribution', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(seeds_list)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 2. 난이도 점수
    ax2 = axes[1]
    scores = [d['difficulty_score'] for d in difficulties]
    colors = ['green' if s < 5 else 'orange' if s < 10 else 'red' for s in scores]
    
    ax2.bar(x, scores, alpha=0.7, color=colors)
    ax2.set_xlabel('Seed', fontsize=12)
    ax2.set_ylabel('Difficulty Score', fontsize=12)
    ax2.set_title('Map Difficulty Comparison', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(seeds_list)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 난이도 범례
    ax2.axhline(y=5, color='green', linestyle='--', alpha=0.5, label='Easy')
    ax2.axhline(y=10, color='orange', linestyle='--', alpha=0.5, label='Medium')
    ax2.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n💾 난이도 비교 저장: {save_path}")
    else:
        plt.show()
    
    plt.close()
    
    print("\n" + "="*60 + "\n")


if __name__ == "__main__":
    """
    메인 실행
    
    사용법:
        # 모든 시드 맵 비교
        python visualize_maps.py
        
        # 특정 시드만 확인
        python visualize_maps.py --seeds 1000 2000
        
        # 난이도 분석
        python visualize_maps.py --difficulty
    """
    import argparse
    from utils.path_utils import get_result_path
    
    parser = argparse.ArgumentParser(description="시드별 맵 시각화")
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                       help="시각화할 시드 리스트")
    parser.add_argument("--difficulty", action="store_true",
                       help="난이도 분석 실행")
    parser.add_argument("--save", action="store_true",
                       help="결과 저장")
    
    args = parser.parse_args()
    
    # 시드 설정
    seeds = args.seeds if args.seeds else [FIXED_SEED] + TEST_SEEDS
    
    # 맵 비교
    save_path = get_result_path("seed_maps_comparison.png") if args.save else None
    compare_all_seeds(seeds, save_path)
    
    # 난이도 분석
    if args.difficulty:
        diff_save_path = get_result_path("seed_difficulty_comparison.png") if args.save else None
        compare_seed_difficulty(seeds, diff_save_path)
