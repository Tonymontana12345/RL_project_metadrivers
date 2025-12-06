"""
진짜 3D 렌더링 스크립트

MetaDrive의 3D 엔진을 사용하여 실제 게임처럼 보이는 화면 캡처
"""

import numpy as np
from metadrive import MetaDriveEnv
from config import FIXED_SEED, TEST_SEEDS
import os
from PIL import Image
import matplotlib.pyplot as plt

def capture_3d_view(seed, num_frames=10, save_dir=None):
    """
    시드별 맵을 3D로 렌더링하여 캡처
    
    Args:
        seed: 맵 생성 시드
        num_frames: 캡처할 프레임 수
        save_dir: 저장 디렉토리
    
    Returns:
        frames: 캡처된 프레임 리스트
    """
    print(f"\n🎮 시드 {seed} 3D 렌더링 중...")
    
    # 환경 생성 (이미지 관찰)
    env = MetaDriveEnv({
        "start_seed": seed,
        "num_scenarios": 1,
        "map": 5,
        "use_render": False,
        "image_observation": True,
        "manual_control": False,
        "traffic_density": 0.1,  # 차량 추가
        "window_size": (1920, 1080),  # Full HD 해상도
        "vehicle_config": {
            "image_source": "rgb_camera",
        },
        "sensors": {
            "rgb_camera": ("MainCamera", 1920, 1080),
        },
    })
    
    try:
        obs, info = env.reset(seed=seed)
        
        frames = []
        
        # 여러 프레임 캡처 (차량이 움직이면서)
        for i in range(num_frames):
            # 자동으로 전진
            action = [0.5, 0]  # [가속, 조향]
            obs, reward, terminated, truncated, info = env.step(action)
            
            # 렌더링 (3D 뷰 - image_observation에서 가져오기)
            if isinstance(obs, dict) and 'image' in obs:
                frame = obs['image']
            else:
                frame = env.render(mode='rgb_array')
            
            if frame is not None:
                frames.append(frame)
                
                # 개별 프레임 저장
                if save_dir and i % 2 == 0:  # 2프레임마다 저장
                    frame_path = os.path.join(save_dir, f"seed_{seed}_frame_{i:02d}.png")
                    Image.fromarray(frame).save(frame_path)
            
            if terminated or truncated:
                break
        
        print(f"✅ {len(frames)}개 프레임 캡처 완료")
        
        env.close()
        return frames
        
    except Exception as e:
        print(f"❌ 3D 렌더링 실패: {e}")
        env.close()
        return []


def create_3d_montage(seed, save_path=None):
    """
    여러 각도의 3D 뷰를 하나의 이미지로 합성
    
    Args:
        seed: 맵 생성 시드
        save_path: 저장 경로
    """
    print(f"\n🎬 시드 {seed} 3D 몽타주 생성 중...")
    
    # 환경 생성 (이미지 관찰)
    env = MetaDriveEnv({
        "start_seed": seed,
        "num_scenarios": 1,
        "map": 5,
        "use_render": False,
        "image_observation": True,
        "manual_control": False,
        "traffic_density": 0.1,
        "window_size": (800, 600),
        "vehicle_config": {
            "image_source": "rgb_camera",
        },
        "sensors": {
            "rgb_camera": ("MainCamera", 800, 600),
        },
    })
    
    try:
        obs, info = env.reset(seed=seed)
        
        # 4개의 다른 시점 캡처
        views = []
        positions = [0, 20, 40, 60]  # 다른 위치
        
        for i, steps in enumerate(positions):
            # 해당 위치까지 이동
            for _ in range(steps):
                action = [0.5, 0]
                obs, reward, terminated, truncated, info = env.step(action)
                if terminated or truncated:
                    env.reset(seed=seed)
                    break
            
            # 렌더링 (image_observation에서 가져오기)
            if isinstance(obs, dict) and 'image' in obs:
                frame = obs['image']
            else:
                frame = env.render(mode='rgb_array')
            
            if frame is not None:
                views.append(frame)
        
        env.close()
        
        # 4개 뷰를 2x2 그리드로 합성
        if len(views) >= 4:
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle(f'Seed {seed} - 3D Views from Different Positions', 
                        fontsize=16, fontweight='bold')
            
            for idx, (ax, view) in enumerate(zip(axes.flat, views)):
                ax.imshow(view)
                ax.set_title(f'Position {positions[idx]} steps', fontsize=12)
                ax.axis('off')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                print(f"💾 3D 몽타주 저장: {save_path}")
            else:
                plt.show()
            
            plt.close()
        else:
            print("⚠️  충분한 뷰를 캡처하지 못했습니다")
            
    except Exception as e:
        print(f"❌ 몽타주 생성 실패: {e}")
        env.close()


def capture_single_3d_frame(seed, save_path=None, steps=10):
    """
    단일 3D 프레임 캡처 (고해상도)
    
    Args:
        seed: 맵 생성 시드
        save_path: 저장 경로
        steps: 시작점에서 몇 스텝 이동할지
    """
    print(f"\n📸 시드 {seed} 3D 스냅샷 캡처 중...")
    
    # 환경 생성 (고해상도, 이미지 관찰)
    env = MetaDriveEnv({
        "start_seed": seed,
        "num_scenarios": 1,
        "map": 5,
        "use_render": False,  # 렌더링 비활성화
        "image_observation": True,  # 이미지 관찰 활성화
        "manual_control": False,
        "traffic_density": 0.1,
        "window_size": (1920, 1080),  # Full HD
        "vehicle_config": {
            "image_source": "rgb_camera",
        },
        "sensors": {
            "rgb_camera": ("MainCamera", 1920, 1080),
        },
    })
    
    try:
        obs, info = env.reset(seed=seed)
        
        # 원하는 위치까지 이동
        for _ in range(steps):
            action = [0.5, 0]  # 전진
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
        
        # 3D 렌더링 (image_observation에서 가져오기)
        if isinstance(obs, dict) and 'image' in obs:
            frame = obs['image']
        else:
            frame = env.render(mode='rgb_array')
        
        if frame is not None and save_path:
            Image.fromarray(frame).save(save_path)
            print(f"✅ 저장 완료: {save_path}")
            print(f"   해상도: {frame.shape[1]}x{frame.shape[0]}")
        
        env.close()
        return frame
        
    except Exception as e:
        print(f"❌ 캡처 실패: {e}")
        env.close()
        return None


def render_all_seeds_3d(seeds=None, save_dir=None, style='montage'):
    """
    모든 시드를 3D로 렌더링
    
    Args:
        seeds: 시드 리스트
        save_dir: 저장 디렉토리
        style: 'montage' (4개 뷰) 또는 'single' (단일 뷰)
    """
    if seeds is None:
        seeds = [FIXED_SEED] + TEST_SEEDS
    
    if save_dir is None:
        from utils.path_utils import get_result_path
        save_dir = os.path.dirname(get_result_path("dummy"))
    
    print("\n" + "="*60)
    print("🎮 3D 렌더링 시작")
    print("="*60)
    print(f"스타일: {style}")
    print(f"시드: {seeds}")
    print("="*60)
    
    for seed in seeds:
        if style == 'montage':
            save_path = os.path.join(save_dir, f"seed_{seed}_3d_montage.png")
            create_3d_montage(seed, save_path)
        else:  # single
            save_path = os.path.join(save_dir, f"seed_{seed}_3d_view.png")
            capture_single_3d_frame(seed, save_path, steps=10)
    
    print("\n" + "="*60)
    print("✅ 3D 렌더링 완료!")
    print(f"📁 저장 위치: {save_dir}")
    print("="*60 + "\n")


def create_comparison_view(seed, save_path=None):
    """
    2D 탑뷰 vs 3D 뷰 비교
    
    Args:
        seed: 맵 생성 시드
        save_path: 저장 경로
    """
    print(f"\n🔄 시드 {seed} 2D vs 3D 비교 생성 중...")
    
    # 환경 생성 (이미지 관찰)
    env = MetaDriveEnv({
        "start_seed": seed,
        "num_scenarios": 1,
        "map": 5,
        "use_render": False,
        "image_observation": True,
        "manual_control": False,
        "window_size": (800, 600),
        "vehicle_config": {
            "image_source": "rgb_camera",
        },
        "sensors": {
            "rgb_camera": ("MainCamera", 800, 600),
        },
    })
    
    try:
        obs, info = env.reset(seed=seed)
        
        # 약간 이동
        for _ in range(20):
            action = [0.5, 0]
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
        
        # 3D 뷰 캡처 (image_observation에서 가져오기)
        if isinstance(obs, dict) and 'image' in obs:
            view_3d = obs['image']
        else:
            view_3d = env.render(mode='rgb_array')
        
        # 탑뷰 캡처 (가능하면)
        try:
            view_top = env.render(mode='topdown')
        except:
            view_top = None
        
        env.close()
        
        # 비교 이미지 생성
        if view_3d is not None:
            if view_top is not None:
                fig, axes = plt.subplots(1, 2, figsize=(16, 6))
                
                axes[0].imshow(view_top)
                axes[0].set_title('Top-Down View (2D)', fontsize=14, fontweight='bold')
                axes[0].axis('off')
                
                axes[1].imshow(view_3d)
                axes[1].set_title('First-Person View (3D)', fontsize=14, fontweight='bold')
                axes[1].axis('off')
                
                fig.suptitle(f'Seed {seed} - View Comparison', 
                           fontsize=16, fontweight='bold')
            else:
                fig, ax = plt.subplots(figsize=(12, 8))
                ax.imshow(view_3d)
                ax.set_title(f'Seed {seed} - 3D View', fontsize=14, fontweight='bold')
                ax.axis('off')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                print(f"💾 비교 이미지 저장: {save_path}")
            else:
                plt.show()
            
            plt.close()
        
    except Exception as e:
        print(f"❌ 비교 뷰 생성 실패: {e}")
        env.close()


if __name__ == "__main__":
    """
    메인 실행
    
    사용법:
        # 모든 시드 3D 몽타주 (4개 뷰)
        python render_3d_view.py
        
        # 단일 3D 뷰
        python render_3d_view.py --style single
        
        # 특정 시드만
        python render_3d_view.py --seeds 1000 2000
        
        # 2D vs 3D 비교
        python render_3d_view.py --compare
    """
    import argparse
    from utils.path_utils import get_result_path
    
    parser = argparse.ArgumentParser(description="3D 렌더링")
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                       help="렌더링할 시드 리스트")
    parser.add_argument("--style", type=str, default="montage",
                       choices=["montage", "single"],
                       help="렌더링 스타일")
    parser.add_argument("--compare", action="store_true",
                       help="2D vs 3D 비교")
    
    args = parser.parse_args()
    
    seeds = args.seeds if args.seeds else [FIXED_SEED] + TEST_SEEDS
    
    if args.compare:
        # 2D vs 3D 비교
        for seed in seeds:
            save_path = get_result_path(f"seed_{seed}_comparison.png")
            create_comparison_view(seed, save_path)
    else:
        # 일반 3D 렌더링
        render_all_seeds_3d(seeds, style=args.style)
