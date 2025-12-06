"""
3D 맵 탐색 스크립트

키보드로 직접 운전하면서 맵을 3D로 탐색할 수 있습니다.
"""

from metadrive import MetaDriveEnv
from config import FIXED_SEED, TEST_SEEDS
import argparse


def view_map_3d(seed, auto_drive=False):
    """
    3D로 맵 탐색
    
    Args:
        seed: 맵 시드
        auto_drive: True면 자동 주행, False면 수동 제어
    """
    print("\n" + "="*60)
    print(f"🎮 시드 {seed} 맵 3D 탐색")
    print("="*60)
    
    if not auto_drive:
        print("\n📋 조작법:")
        print("  W / ↑  : 가속")
        print("  S / ↓  : 감속/후진")
        print("  A / ←  : 좌회전")
        print("  D / →  : 우회전")
        print("  Q      : 종료")
        print("  마우스  : 시점 회전")
        print("\n💡 Tip: 천천히 운전하면서 맵 구조를 확인하세요!")
    else:
        print("\n🚗 자동 주행 모드")
        print("  ESC 또는 창 닫기로 종료")
    
    print("="*60 + "\n")
    
    # 환경 생성
    env = MetaDriveEnv({
        "start_seed": seed,
        "num_scenarios": 1,
        "map": 5,  # 5개 블록
        "use_render": True,           # 3D 렌더링 활성화
        "manual_control": not auto_drive,  # 수동/자동 선택
        "image_observation": False,   # 센서 비활성화
        "traffic_density": 0.1,       # 다른 차량 추가
        "show_fps": True,             # FPS 표시
        "show_interface": True,       # 인터페이스 표시
        "show_logo": False,           # 로고 숨김
        "show_skybox": True,          # 하늘 표시
        "daytime": "08:00",           # 낮 시간
    })
    
    try:
        env.reset(seed=seed)
        
        if not auto_drive:
            # 수동 제어 모드
            print("🎮 수동 제어 시작! (Q를 눌러 종료)")
            env.run()  # 사용자가 직접 운전
        else:
            # 자동 주행 모드
            print("🚗 자동 주행 시작! (ESC로 종료)")
            for i in range(2000):  # 충분히 긴 시간
                # 직진
                action = [0.5, 0]  # [가속, 조향]
                obs, reward, terminated, truncated, info = env.step(action)
                
                if terminated or truncated:
                    print(f"\n에피소드 종료: {i} 스텝")
                    if info.get("arrive_dest", False):
                        print("✅ 목적지 도착!")
                    elif info.get("crash", False):
                        print("💥 충돌!")
                    elif info.get("out_of_road", False):
                        print("🚧 도로 이탈!")
                    break
        
        env.close()
        print("\n✅ 탐색 완료!")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  사용자가 중단했습니다.")
        env.close()
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        env.close()


def view_all_seeds(auto_drive=False):
    """
    모든 시드를 순차적으로 탐색
    
    Args:
        auto_drive: True면 자동 주행, False면 수동 제어
    """
    seeds = [FIXED_SEED] + TEST_SEEDS
    
    print("\n" + "="*60)
    print("🗺️  모든 시드 순차 탐색")
    print("="*60)
    print(f"시드: {seeds}")
    print("="*60 + "\n")
    
    for i, seed in enumerate(seeds, 1):
        print(f"\n[{i}/{len(seeds)}] 시드 {seed} 탐색 중...")
        view_map_3d(seed, auto_drive)
        
        if i < len(seeds):
            input("\n⏸️  다음 시드로 이동하려면 Enter를 누르세요...")


if __name__ == "__main__":
    """
    메인 실행
    
    사용법:
        # 특정 시드 수동 제어
        python view_map_3d.py --seed 1000
        
        # 특정 시드 자동 주행
        python view_map_3d.py --seed 1000 --auto
        
        # 모든 시드 순차 탐색 (수동)
        python view_map_3d.py --all
        
        # 모든 시드 순차 탐색 (자동)
        python view_map_3d.py --all --auto
    """
    parser = argparse.ArgumentParser(description="3D 맵 탐색")
    parser.add_argument("--seed", type=int, default=None,
                       help="탐색할 시드 (기본: 1000)")
    parser.add_argument("--all", action="store_true",
                       help="모든 시드 순차 탐색")
    parser.add_argument("--auto", action="store_true",
                       help="자동 주행 모드")
    
    args = parser.parse_args()
    
    if args.all:
        # 모든 시드 탐색
        view_all_seeds(auto_drive=args.auto)
    else:
        # 단일 시드 탐색
        seed = args.seed if args.seed else FIXED_SEED
        view_map_3d(seed, auto_drive=args.auto)
