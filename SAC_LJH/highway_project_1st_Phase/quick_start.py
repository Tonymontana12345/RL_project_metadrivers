"""
빠른 시작 가이드

PGDrive 환경을 체험하고 기본 사용법을 익힙니다.
"""

import gymnasium as gym
import numpy as np
from config import FIXED_SEED, FIXED_SEED_ENV_CONFIG
from envs.metadrive_env import make_env


def demo_environment():
    """환경 기본 사용법 데모"""
    print("\n" + "="*60)
    print("🚗 PGDrive 환경 데모")
    print("="*60 + "\n")
    
    # 환경 생성
    print("1️⃣  환경 생성")
    env = make_env(seed=FIXED_SEED, render=False)()
    print(f"   ✅ 환경 생성 완료 (시드: {FIXED_SEED})\n")
    
    # 관측 및 액션 공간
    print("2️⃣  관측 및 액션 공간")
    print(f"   관측 공간: {env.observation_space}")
    print(f"   액션 공간: {env.action_space}\n")
    
    # 에피소드 실행
    print("3️⃣  랜덤 에피소드 실행")
    obs, info = env.reset()
    print(f"   초기 관측 shape: {obs.shape}")
    
    done = False
    total_reward = 0
    steps = 0
    
    while not done and steps < 1000:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        steps += 1
    
    print(f"   총 보상: {total_reward:.2f}")
    print(f"   스텝 수: {steps}")
    print(f"   성공: {info.get('arrive_dest', False)}")
    print(f"   충돌: {info.get('crash', False)}\n")
    
    env.close()
    print("✅ 데모 완료!\n")


def manual_control_demo():
    """수동 제어 데모"""
    print("\n" + "="*60)
    print("🎮 수동 제어 데모")
    print("="*60)
    print("\n키보드 조작:")
    print("  W/S: 가속/감속")
    print("  A/D: 좌회전/우회전")
    print("  ESC: 종료\n")
    print("환경을 실행합니다...\n")
    
    env = make_env(seed=FIXED_SEED, render=True)()
    
    obs, info = env.reset()
    
    try:
        for _ in range(10000):
            obs, reward, terminated, truncated, info = env.step([0, 0])  # 수동 제어 시 액션 무시됨
            done = terminated or truncated
            env.render()
            
            if done:
                print(f"\n에피소드 종료!")
                print(f"  성공: {info.get('arrive_dest', False)}")
                print(f"  충돌: {info.get('crash', False)}")
                obs, info = env.reset()
    
    except KeyboardInterrupt:
        print("\n\n수동 제어 종료")
    
    env.close()


def test_different_seeds():
    """다양한 시드 테스트"""
    print("\n" + "="*60)
    print("🎲 다양한 시드 테스트")
    print("="*60 + "\n")
    
    test_seeds = [1000, 2000, 3000]
    
    for seed in test_seeds:
        print(f"시드 {seed} 테스트:")
        
        env = make_env(seed=seed, render=False)()
        
        obs, info = env.reset()
        done = False
        total_reward = 0
        steps = 0
        
        while not done and steps < 1000:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            steps += 1
        
        print(f"  보상: {total_reward:.2f}, 스텝: {steps}, 성공: {info.get('arrive_dest', False)}\n")
        
        env.close()


def test_fixed_vs_random_traffic():
    """고정 트래픽 vs 랜덤 트래픽 비교"""
    print("\n" + "="*60)
    print("🚦 고정 트래픽 vs 랜덤 트래픽")
    print("="*60 + "\n")
    
    configs = [
        {"random_traffic": False, "name": "고정 트래픽"},
        {"random_traffic": True, "name": "랜덤 트래픽"},
    ]
    
    for config_type in configs:
        print(f"{config_type['name']} 테스트:")
        
        env = make_env(seed=FIXED_SEED, render=False)()
        
        # 2번 실행해서 재현성 확인
        rewards = []
        for run in range(2):
            obs, info = env.reset()
            done = False
            total_reward = 0
            
            while not done:
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                total_reward += reward
            
            rewards.append(total_reward)
        
        print(f"  Run 1: {rewards[0]:.2f}")
        print(f"  Run 2: {rewards[1]:.2f}")
        print(f"  동일: {abs(rewards[0] - rewards[1]) < 0.01}\n")
        
        env.close()


def show_map_types():
    """다양한 맵 타입 보기"""
    print("\n" + "="*60)
    print("🗺️  다양한 맵 타입")
    print("="*60 + "\n")
    
    map_configs = [
        {"map": 3, "name": "3개 블록"},
        # {"map": 5, "name": "5개 블록"},
        # {"map": "SCSCS", "name": "커스텀 (Straight-Circular-Straight-Circular-Straight)"},
        # {"map": "XOXO", "name": "커스텀 (Intersection-Roundabout-Intersection-Roundabout)"},
    ]
    
    for map_config in map_configs:
        print(f"{map_config['name']}:")
        
        try:
            env = make_env(seed=FIXED_SEED, render=False)()
            
            obs, info = env.reset()
            print(f"  ✅ 생성 성공 (관측 shape: {obs.shape})")
            env.close()
        
        except Exception as e:
            print(f"  ❌ 생성 실패: {e}")
        
        print()


if __name__ == "__main__":
    """
    메인 실행
    
    사용법:
        # 전체 데모
        python quick_start.py
        
        # 수동 제어만
        python quick_start.py --manual
        
        # 환경 데모만
        python quick_start.py --demo
    """
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--manual":
            manual_control_demo()
        elif sys.argv[1] == "--demo":
            demo_environment()
        else:
            print(f"❌ 알 수 없는 옵션: {sys.argv[1]}")
            print("사용법: python quick_start.py [--demo | --manual]")
    
    else:
        # 전체 데모
        print("\n" + "="*60)
        print("🎓 PGDrive 빠른 시작 가이드")
        print("="*60)
        
        demo_environment()
        test_different_seeds()
        test_fixed_vs_random_traffic()
        show_map_types()
        
        print("\n" + "="*60)
        print("✅ 모든 데모 완료!")
        print("="*60)
        print("\n다음 단계:")
        print("  1. python train_fixed_seed.py --quick  # 빠른 학습 테스트")
        print("  2. python train_fixed_seed.py          # 본격 학습")
        print("  3. python evaluate.py                  # 평가")
        print("  4. python visualize.py                 # 시각화")
        print("\n수동 제어 체험:")
        print("  python quick_start.py --manual\n")
