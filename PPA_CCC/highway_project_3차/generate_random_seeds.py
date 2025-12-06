"""
랜덤 시드 생성 스크립트

학습용 시드와 평가용 시드를 랜덤하게 선정합니다.
재현성을 위해 고정 시드를 사용합니다.
"""

import random
import argparse


def generate_seeds(n_train=3, n_test=5, seed=42, min_seed=1000, max_seed=9999):
    """
    랜덤 시드 생성
    
    Args:
        n_train: 학습용 시드 개수
        n_test: 평가용 시드 개수
        seed: 랜덤 생성용 시드 (재현성)
        min_seed: 최소 시드 값
        max_seed: 최대 시드 값
    
    Returns:
        train_seeds, test_seeds
    """
    # 재현성을 위한 시드 설정
    random.seed(seed)
    
    # 전체 시드 풀
    all_seeds = list(range(min_seed, max_seed + 1))
    
    # 랜덤하게 선정 (중복 없음)
    total_needed = n_train + n_test
    selected_seeds = random.sample(all_seeds, total_needed)
    
    # 학습용, 평가용으로 분리
    train_seeds = sorted(selected_seeds[:n_train])
    test_seeds = sorted(selected_seeds[n_train:])
    
    return train_seeds, test_seeds


def main():
    parser = argparse.ArgumentParser(description="랜덤 시드 생성")
    
    parser.add_argument(
        "--train",
        type=int,
        default=3,
        help="학습용 시드 개수 (기본: 3)"
    )
    
    parser.add_argument(
        "--test",
        type=int,
        default=5,
        help="평가용 시드 개수 (기본: 5)"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="랜덤 생성용 시드 (재현성) (기본: 42)"
    )
    
    parser.add_argument(
        "--min",
        type=int,
        default=1000,
        help="최소 시드 값 (기본: 1000)"
    )
    
    parser.add_argument(
        "--max",
        type=int,
        default=9999,
        help="최대 시드 값 (기본: 9999)"
    )
    
    args = parser.parse_args()
    
    # 시드 생성
    train_seeds, test_seeds = generate_seeds(
        n_train=args.train,
        n_test=args.test,
        seed=args.seed,
        min_seed=args.min,
        max_seed=args.max
    )
    
    # 결과 출력
    print("\n" + "="*60)
    print("🎲 랜덤 시드 생성 (재현 가능)")
    print("="*60)
    print(f"\n생성 조건:")
    print(f"  - 학습용 시드: {args.train}개")
    print(f"  - 평가용 시드: {args.test}개")
    print(f"  - 랜덤 시드: {args.seed} (재현용)")
    print(f"  - 시드 범위: {args.min}-{args.max}")
    
    print("\n" + "-"*60)
    print(f"📚 학습용 시드 ({len(train_seeds)}개):")
    print(f"TRAIN_SEEDS = {train_seeds}")
    
    print(f"\n📊 평가용 시드 ({len(test_seeds)}개):")
    print(f"TEST_SEEDS = {test_seeds}")
    
    print("\n" + "-"*60)
    print("📝 config.py에 다음 내용을 복사하세요:")
    print("-"*60)
    print(f"\n# 학습용 시드 (랜덤 선정, 고정)")
    print(f"TRAIN_SEEDS = {train_seeds}")
    print(f"\n# 평가용 시드 (랜덤 선정, 고정)")
    print(f"TEST_SEEDS = {test_seeds}")
    
    print("\n" + "="*60)
    print("✅ 시드 생성 완료!")
    print("="*60)
    
    # 추가 정보
    print("\n💡 사용법:")
    print("  1. 위의 시드를 config.py에 복사")
    print("  2. python train.py --mode multi --algorithm ppo")
    print("  3. python evaluate.py --model <모델경로>")
    
    print("\n⚠️  주의:")
    print("  - 이 시드들은 고정되어 재현 가능합니다")
    print("  - 다른 시드를 원하면 --seed 값을 변경하세요")
    print(f"    예: python {__file__} --seed 123\n")


if __name__ == "__main__":
    """
    사용법:
    
    # 기본 (학습 3개, 평가 5개)
    python generate_random_seeds.py
    
    # 학습 5개, 평가 5개
    python generate_random_seeds.py --train 5 --test 5
    
    # 다른 랜덤 시드로 생성
    python generate_random_seeds.py --seed 123
    
    # 시드 범위 변경
    python generate_random_seeds.py --min 5000 --max 9999
    """
    main()

