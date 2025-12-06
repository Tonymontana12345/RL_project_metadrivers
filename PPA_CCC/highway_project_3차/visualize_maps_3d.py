"""
시드별 맵 3D 시각화 스크립트

실제 도로처럼 보이는 3D 렌더링 및 탑뷰 캡처
"""

import matplotlib.pyplot as plt
import numpy as np
from metadrive import MetaDriveEnv
from config import FIXED_SEED, TEST_SEEDS
import os
from PIL import Image

def capture_map_topdown(seed, save_path=None, resolution=(800, 800)):
    """
    시드별 맵을 탑뷰로 캡처
    
    Args:
        seed: 맵 생성 시드
        save_path: 저장 경로
        resolution: 이미지 해상도
    
    Returns:
        image: 캡처된 이미지 (numpy array)
    """
    # 환경 생성 (렌더링 활성화)
    env = MetaDriveEnv({
        "start_seed": seed,
        "num_scenarios": 1,
        "map": 5,
        "use_render": True,  # 렌더링 활성화
        "manual_control": False,
        "offscreen_render": True,  # 오프스크린 렌더링
        "image_observation": True,  # 이미지 관찰
        "window_size": resolution,
    })
    
    try:
        # 환경 리셋
        obs, info = env.reset(seed=seed)
        
        # 탑뷰 이미지 캡처
        # MetaDrive의 탑뷰 렌더링 사용
        if hasattr(env, 'render'):
            # 여러 스텝 진행하여 맵 전체 보기
            for _ in range(10):
                action = [0, 0]  # 정지 상태
                obs, reward, terminated, truncated, info = env.step(action)
            
            # 렌더링
            img = env.render(mode='topdown')
            
            if save_path:
                if isinstance(img, np.ndarray):
                    Image.fromarray(img).save(save_path)
                    print(f"💾 맵 이미지 저장: {save_path}")
            
            return img
        else:
            print("⚠️  렌더링 기능을 사용할 수 없습니다")
            return None
            
    except Exception as e:
        print(f"❌ 맵 캡처 실패: {e}")
        return None
    finally:
        env.close()


def capture_map_birdseye(seed, save_path=None):
    """
    시드별 맵을 조감도(bird's eye view)로 캡처
    
    Args:
        seed: 맵 생성 시드
        save_path: 저장 경로
    """
    # 환경 생성
    env = MetaDriveEnv({
        "start_seed": seed,
        "num_scenarios": 1,
        "map": 5,
        "use_render": True,
        "manual_control": False,
    })
    
    try:
        env.reset(seed=seed)
        
        # 맵 정보 추출
        current_map = env.current_map
        
        # matplotlib로 도로 네트워크 그리기
        fig, ax = plt.subplots(figsize=(12, 12))
        
        # 도로 그리기
        if hasattr(current_map, 'road_network'):
            road_network = current_map.road_network
            
            # 모든 차선 그리기
            for road_id, road in road_network.graph.items():
                if hasattr(road, 'lanes'):
                    for lane_idx, lane in enumerate(road.lanes):
                        # 차선의 중심선 가져오기
                        if hasattr(lane, 'position'):
                            # 차선을 따라 점들을 샘플링
                            points = []
                            for s in np.linspace(0, lane.length, 100):
                                try:
                                    pos = lane.position(s, 0)
                                    points.append(pos)
                                except:
                                    continue
                            
                            if points:
                                points = np.array(points)
                                # 차선 그리기
                                ax.plot(points[:, 0], points[:, 1], 
                                       'gray', linewidth=3, alpha=0.7)
                                
                                # 차선 경계 그리기
                                left_points = []
                                right_points = []
                                for s in np.linspace(0, lane.length, 50):
                                    try:
                                        left = lane.position(s, lane.width_at(s) / 2)
                                        right = lane.position(s, -lane.width_at(s) / 2)
                                        left_points.append(left)
                                        right_points.append(right)
                                    except:
                                        continue
                                
                                if left_points and right_points:
                                    left_points = np.array(left_points)
                                    right_points = np.array(right_points)
                                    ax.plot(left_points[:, 0], left_points[:, 1], 
                                           'white', linewidth=1, linestyle='--', alpha=0.5)
                                    ax.plot(right_points[:, 0], right_points[:, 1], 
                                           'white', linewidth=1, linestyle='--', alpha=0.5)
        
        # 시작점과 목적지 표시
        ax.plot(0, 0, 'go', markersize=20, label='Start', zorder=10)
        
        # 스타일 설정
        ax.set_facecolor('#2d5016')  # 잔디 색
        ax.set_aspect('equal')
        ax.grid(False)
        ax.legend(fontsize=14, loc='upper right')
        ax.set_title(f'Seed {seed} - Bird\'s Eye View', 
                    fontsize=16, fontweight='bold', pad=20)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight', 
                       facecolor='#2d5016')
            print(f"💾 조감도 저장: {save_path}")
        else:
            plt.show()
        
        plt.close()
        
    except Exception as e:
        print(f"❌ 조감도 생성 실패: {e}")
    finally:
        env.close()


def visualize_map_realistic(seed, save_path=None):
    """
    실제 도로처럼 보이는 맵 시각화
    
    Args:
        seed: 맵 생성 시드
        save_path: 저장 경로
    """
    print(f"\n🗺️  시드 {seed} 맵 시각화 중...")
    
    # 환경 생성
    env = MetaDriveEnv({
        "start_seed": seed,
        "num_scenarios": 1,
        "map": 5,
        "use_render": False,
        "manual_control": False,
    })
    
    obs, info = env.reset(seed=seed)
    
    # 맵 정보 추출
    current_map = env.current_map
    
    # 큰 캔버스 생성
    fig, ax = plt.subplots(figsize=(16, 16))
    
    # 배경색 (잔디)
    ax.set_facecolor('#3a5f0b')
    
    # 도로 네트워크 그리기
    if hasattr(current_map, 'road_network'):
        road_network = current_map.road_network
        
        # 모든 도로 수집
        all_lanes = []
        
        # road_network의 모든 도로와 차선 가져오기
        if hasattr(road_network, 'graph'):
            for road_key, road_dict in road_network.graph.items():
                # NodeRoadNetwork의 경우 road_dict는 dict 타입
                if isinstance(road_dict, dict):
                    # dict의 값들이 lane 객체
                    for lane_key, lane in road_dict.items():
                        if lane is not None:
                            all_lanes.append(lane)
                # Road 객체에서 lanes 가져오기
                elif hasattr(road_dict, 'lanes'):
                    for lane in road_dict.lanes:
                        all_lanes.append(lane)
                # 또는 직접 Road가 Lane일 수도 있음
                elif hasattr(road_dict, 'position'):
                    all_lanes.append(road_dict)
        
        print(f"  총 {len(all_lanes)}개 차선 발견")
        
        # 각 차선 그리기
        for lane in all_lanes:
            try:
                # lane이 position 메서드를 가지고 있는지 확인
                if not hasattr(lane, 'position'):
                    continue
                
                # length 가져오기
                if hasattr(lane, 'length'):
                    length = lane.length
                elif hasattr(lane, 'get_length'):
                    length = lane.get_length()
                else:
                    length = 50  # 기본값
                
                # 차선 샘플링
                num_points = 100
                
                # 도로 표면 (어두운 회색)
                left_edge = []
                right_edge = []
                center_line = []
                
                for s in np.linspace(0, length, num_points):
                    try:
                        # 차선 너비
                        if hasattr(lane, 'width_at'):
                            width = lane.width_at(s)
                        elif hasattr(lane, 'width'):
                            width = lane.width
                        else:
                            width = 3.5
                        
                        # 위치 계산
                        left = lane.position(s, width / 2)
                        right = lane.position(s, -width / 2)
                        center = lane.position(s, 0)
                        
                        left_edge.append([left[0], left[1]])
                        right_edge.append([right[0], right[1]])
                        center_line.append([center[0], center[1]])
                    except Exception as e:
                        continue
                
                if len(left_edge) > 10 and len(right_edge) > 10:
                    left_edge = np.array(left_edge)
                    right_edge = np.array(right_edge)
                    center_line = np.array(center_line)
                    
                    # 도로 표면 채우기
                    vertices = np.vstack([left_edge, right_edge[::-1]])
                    from matplotlib.patches import Polygon
                    road_polygon = Polygon(vertices, 
                                          facecolor='#3d3d3d', 
                                          edgecolor='none',
                                          alpha=0.9,
                                          zorder=1)
                    ax.add_patch(road_polygon)
                    
                    # 차선 중앙선 (노란색 점선)
                    ax.plot(center_line[:, 0], center_line[:, 1],
                           color='#ffd700', linewidth=2, 
                           linestyle='--', alpha=0.8, zorder=2)
                    
                    # 도로 가장자리 (흰색 실선)
                    ax.plot(left_edge[:, 0], left_edge[:, 1],
                           color='white', linewidth=3, 
                           linestyle='-', alpha=0.9, zorder=3)
                    ax.plot(right_edge[:, 0], right_edge[:, 1],
                           color='white', linewidth=3, 
                           linestyle='-', alpha=0.9, zorder=3)
            except Exception as e:
                print(f"  차선 그리기 오류: {e}")
                continue
    
    # 시작점 (초록 원)
    ax.plot(0, 0, 'o', color='#00ff00', markersize=25, 
           label='Start', zorder=10, markeredgecolor='white', 
           markeredgewidth=2)
    
    # 목적지 추정 위치 (빨간 별)
    # 맵의 대략적인 끝 지점
    if hasattr(current_map, 'road_network'):
        max_x, max_y = 0, 0
        for road_id, road in road_network.graph.items():
            if hasattr(road, 'lanes'):
                for lane in road.lanes:
                    if hasattr(lane, 'position') and hasattr(lane, 'length'):
                        try:
                            end_pos = lane.position(lane.length, 0)
                            if abs(end_pos[0]) > abs(max_x):
                                max_x = end_pos[0]
                            if abs(end_pos[1]) > abs(max_y):
                                max_y = end_pos[1]
                        except:
                            continue
        
        if max_x != 0 or max_y != 0:
            ax.plot(max_x, max_y, '*', color='#ff0000', markersize=30,
                   label='Goal (approx)', zorder=10, 
                   markeredgecolor='white', markeredgewidth=2)
    
    # 축 범위 자동 조정 (도로에 맞게)
    ax.autoscale(enable=True, tight=True)
    
    # 축 설정
    ax.set_aspect('equal')
    ax.grid(False)
    ax.set_xlabel('X Position (m)', fontsize=14, color='white')
    ax.set_ylabel('Y Position (m)', fontsize=14, color='white')
    ax.tick_params(colors='white')
    
    # 범례
    ax.legend(fontsize=14, loc='upper right', 
             facecolor='#2d2d2d', edgecolor='white',
             labelcolor='white')
    
    # 제목
    ax.set_title(f'Seed {seed} - Realistic Road View', 
                fontsize=18, fontweight='bold', 
                color='white', pad=20)
    
    # 테두리 색상
    for spine in ax.spines.values():
        spine.set_edgecolor('white')
        spine.set_linewidth(2)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight',
                   facecolor='#3a5f0b')
        print(f"✅ 저장 완료: {save_path}")
    else:
        plt.show()
    
    plt.close()
    env.close()


def compare_all_seeds_realistic(seeds=None, save_dir=None):
    """
    모든 시드를 실제 도로처럼 시각화
    
    Args:
        seeds: 시드 리스트
        save_dir: 저장 디렉토리
    """
    if seeds is None:
        seeds = [FIXED_SEED] + TEST_SEEDS
    
    if save_dir is None:
        from utils.path_utils import get_result_path
        save_dir = os.path.dirname(get_result_path("dummy"))
    
    print("\n" + "="*60)
    print("🗺️  실제 도로 스타일 맵 시각화")
    print("="*60)
    
    for seed in seeds:
        save_path = os.path.join(save_dir, f"seed_{seed}_realistic.png")
        visualize_map_realistic(seed, save_path)
    
    print("\n" + "="*60)
    print("✅ 모든 맵 시각화 완료!")
    print(f"📁 저장 위치: {save_dir}")
    print("="*60 + "\n")


if __name__ == "__main__":
    """
    메인 실행
    
    사용법:
        # 모든 시드 실제 도로 스타일로 시각화
        python visualize_maps_3d.py
        
        # 특정 시드만
        python visualize_maps_3d.py --seeds 1000 2000
        
        # 조감도 스타일
        python visualize_maps_3d.py --style birdseye
    """
    import argparse
    from utils.path_utils import get_result_path
    
    parser = argparse.ArgumentParser(description="실제 도로 스타일 맵 시각화")
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                       help="시각화할 시드 리스트")
    parser.add_argument("--style", type=str, default="realistic",
                       choices=["realistic", "birdseye"],
                       help="시각화 스타일")
    
    args = parser.parse_args()
    
    seeds = args.seeds if args.seeds else [FIXED_SEED] + TEST_SEEDS
    
    if args.style == "realistic":
        compare_all_seeds_realistic(seeds)
    elif args.style == "birdseye":
        for seed in seeds:
            save_path = get_result_path(f"seed_{seed}_birdseye.png")
            capture_map_birdseye(seed, save_path)
