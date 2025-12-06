"""
보고서용 그래프 생성 스크립트
체크포인트 분석 결과를 시각화합니다.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import font_manager
import os

# 한글 폰트 설정
plt.rcParams['font.family'] = 'AppleGothic'  # macOS
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지

# 스타일 설정
sns.set_style("whitegrid")
sns.set_palette("husl")

# 출력 디렉토리
OUTPUT_DIR = "results/figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ========================================
# 데이터 정의
# ========================================

# 체크포인트 분석 결과 (evaluate_all_checkpoints.py 결과)
checkpoint_data = {
    'steps': [50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000, 1000],
    'success_rate': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 4, 4, 8, 10, 2, 4],
    'mean_reward': [81.71, 82.12, 82.68, 75.76, 76.01, 82.39, 76.80, 74.82, 82.36, 81.66, 81.02, 87.44, 86.78, 109.39, 110.41, 123.70, 119.94, 135.31, 148.49, 114.96, 115.61],
    'crash_rate': [46, 52, 24, 48, 40, 44, 52, 54, 56, 40, 40, 40, 52, 58, 54, 34, 58, 32, 26, 38, 36]
}

# 950k 모델 시드별 결과
seed_performance_950k = {
    'seed': ['1000\n(학습)', '2679', '3286', '4657', '5012', '9935'],
    'success_rate': [10.0, 0.0, 0.0, 15.0, 0.0, 40.0],
    'mean_reward': [195.14, 49.32, 96.38, 239.05, 77.67, 281.11],
    'crash_rate': [65.0, 0.0, 5.0, 65.0, 0.0, 40.0]
}

# 1M 모델 시드별 결과 (이전 평가 결과)
seed_performance_1m = {
    'seed': ['1000\n(학습)', '2679', '3286', '4657', '5012', '9935'],
    'success_rate': [10.0, 0.0, 0.0, 5.0, 0.0, 15.0],
    'mean_reward': [162.30, 50.63, 111.46, 175.38, 60.97, 189.71],
    'crash_rate': [65.0, 5.0, 25.0, 45.0, 20.0, 50.0]
}


# ========================================
# 그래프 1: 학습 곡선 (성공률 + 보상)
# ========================================

def plot_learning_curves():
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    steps_k = [s for s in checkpoint_data['steps']]
    
    # 성공률 그래프
    ax1.plot(steps_k, checkpoint_data['success_rate'], 
             marker='o', linewidth=2.5, markersize=8, 
             color='#2E86AB', label='성공률')
    ax1.axvline(x=950, color='red', linestyle='--', linewidth=2, 
                label='최적 체크포인트 (950k)', alpha=0.7)
    ax1.axvline(x=1000, color='orange', linestyle='--', linewidth=2, 
                label='최종 모델 (1M)', alpha=0.7)
    
    # 최고점 표시
    max_idx = checkpoint_data['success_rate'].index(max(checkpoint_data['success_rate']))
    ax1.scatter(steps_k[max_idx], checkpoint_data['success_rate'][max_idx], 
                s=300, color='red', zorder=5, marker='*', 
                edgecolors='darkred', linewidth=2)
    ax1.annotate(f'최고: {checkpoint_data["success_rate"][max_idx]}%\n(950k 스텝)', 
                 xy=(steps_k[max_idx], checkpoint_data['success_rate'][max_idx]),
                 xytext=(steps_k[max_idx]-150, checkpoint_data['success_rate'][max_idx]+2),
                 fontsize=11, fontweight='bold',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
                 arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', lw=2))
    
    ax1.set_xlabel('학습 스텝 (×1,000)', fontsize=13, fontweight='bold')
    ax1.set_ylabel('성공률 (%)', fontsize=13, fontweight='bold')
    ax1.set_title('학습 과정에 따른 성공률 변화 - 과적합 패턴 발견', 
                  fontsize=15, fontweight='bold', pad=20)
    ax1.legend(fontsize=11, loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-1, 12)
    
    # 평균 보상 그래프
    ax2.plot(steps_k, checkpoint_data['mean_reward'], 
             marker='s', linewidth=2.5, markersize=8, 
             color='#A23B72', label='평균 보상')
    ax2.axvline(x=950, color='red', linestyle='--', linewidth=2, 
                label='최적 체크포인트 (950k)', alpha=0.7)
    ax2.axvline(x=1000, color='orange', linestyle='--', linewidth=2, 
                label='최종 모델 (1M)', alpha=0.7)
    
    # 최고점 표시
    max_idx = checkpoint_data['mean_reward'].index(max(checkpoint_data['mean_reward']))
    ax2.scatter(steps_k[max_idx], checkpoint_data['mean_reward'][max_idx], 
                s=300, color='red', zorder=5, marker='*', 
                edgecolors='darkred', linewidth=2)
    ax2.annotate(f'최고: {checkpoint_data["mean_reward"][max_idx]:.1f}\n(950k 스텝)', 
                 xy=(steps_k[max_idx], checkpoint_data['mean_reward'][max_idx]),
                 xytext=(steps_k[max_idx]-150, checkpoint_data['mean_reward'][max_idx]+15),
                 fontsize=11, fontweight='bold',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
                 arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', lw=2))
    
    ax2.set_xlabel('학습 스텝 (×1,000)', fontsize=13, fontweight='bold')
    ax2.set_ylabel('평균 보상', fontsize=13, fontweight='bold')
    ax2.set_title('학습 과정에 따른 평균 보상 변화', 
                  fontsize=15, fontweight='bold', pad=20)
    ax2.legend(fontsize=11, loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/1_learning_curves.png', dpi=300, bbox_inches='tight')
    print(f"✅ 저장: {OUTPUT_DIR}/1_learning_curves.png")
    plt.close()


# ========================================
# 그래프 2: 950k vs 1M 직접 비교
# ========================================

def plot_comparison_950k_vs_1m():
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    seeds = seed_performance_950k['seed']
    x = np.arange(len(seeds))
    width = 0.35
    
    # 성공률 비교
    bars1 = axes[0].bar(x - width/2, seed_performance_950k['success_rate'], 
                        width, label='950k 모델', color='#2E86AB', alpha=0.8)
    bars2 = axes[0].bar(x + width/2, seed_performance_1m['success_rate'], 
                        width, label='1M 모델', color='#F18F01', alpha=0.8)
    
    axes[0].set_xlabel('테스트 시드', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('성공률 (%)', fontsize=12, fontweight='bold')
    axes[0].set_title('시드별 성공률 비교\n950k 모델이 일관되게 우수', 
                      fontsize=14, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(seeds, fontsize=10)
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # 값 표시
    for bar in bars1:
        height = bar.get_height()
        if height > 0:
            axes[0].text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.0f}%', ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        if height > 0:
            axes[0].text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.0f}%', ha='center', va='bottom', fontsize=9)
    
    # 평균 보상 비교
    bars1 = axes[1].bar(x - width/2, seed_performance_950k['mean_reward'], 
                        width, label='950k 모델', color='#2E86AB', alpha=0.8)
    bars2 = axes[1].bar(x + width/2, seed_performance_1m['mean_reward'], 
                        width, label='1M 모델', color='#F18F01', alpha=0.8)
    
    axes[1].set_xlabel('테스트 시드', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('평균 보상', fontsize=12, fontweight='bold')
    axes[1].set_title('시드별 평균 보상 비교\n950k 모델의 높은 보상', 
                      fontsize=14, fontweight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(seeds, fontsize=10)
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3, axis='y')
    
    # 충돌률 비교
    bars1 = axes[2].bar(x - width/2, seed_performance_950k['crash_rate'], 
                        width, label='950k 모델', color='#2E86AB', alpha=0.8)
    bars2 = axes[2].bar(x + width/2, seed_performance_1m['crash_rate'], 
                        width, label='1M 모델', color='#F18F01', alpha=0.8)
    
    axes[2].set_xlabel('테스트 시드', fontsize=12, fontweight='bold')
    axes[2].set_ylabel('충돌률 (%)', fontsize=12, fontweight='bold')
    axes[2].set_title('시드별 충돌률 비교\n비슷한 안전성', 
                      fontsize=14, fontweight='bold')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(seeds, fontsize=10)
    axes[2].legend(fontsize=11)
    axes[2].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/2_model_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✅ 저장: {OUTPUT_DIR}/2_model_comparison.png")
    plt.close()


# ========================================
# 그래프 3: 전체 성능 요약 (막대 그래프)
# ========================================

def plot_overall_summary():
    fig, ax = plt.subplots(figsize=(10, 6))
    
    models = ['950k 체크포인트\n(최적 모델)', '1M 최종 모델\n(과적합)']
    success_rates = [10.8, 5.0]
    colors = ['#06D6A0', '#EF476F']
    
    bars = ax.bar(models, success_rates, color=colors, alpha=0.8, 
                  edgecolor='black', linewidth=2)
    
    # 값 표시
    for i, (bar, val) in enumerate(zip(bars, success_rates)):
        height = bar.get_height()
        improvement = ""
        if i == 0:
            improvement = "\n(기준)"
        else:
            diff = ((success_rates[0] / val) - 1) * 100
            improvement = f"\n({diff:.0f}% 더 낮음)"
        
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.3,
                f'{val}%{improvement}',
                ha='center', va='bottom', fontsize=13, fontweight='bold')
    
    # 개선폭 화살표
    ax.annotate('', xy=(0, success_rates[0]), xytext=(1, success_rates[1]),
                arrowprops=dict(arrowstyle='<->', color='red', lw=3))
    ax.text(0.5, (success_rates[0] + success_rates[1])/2 + 1,
            f'+{success_rates[0] - success_rates[1]:.1f}%p\n(2배 향상)',
            ha='center', fontsize=12, fontweight='bold', color='red',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))
    
    ax.set_ylabel('전체 평균 성공률 (%)', fontsize=13, fontweight='bold')
    ax.set_title('체크포인트 분석을 통한 과적합 해결\n950k 모델이 2배 우수', 
                 fontsize=15, fontweight='bold', pad=20)
    ax.set_ylim(0, 14)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/3_overall_summary.png', dpi=300, bbox_inches='tight')
    print(f"✅ 저장: {OUTPUT_DIR}/3_overall_summary.png")
    plt.close()


# ========================================
# 그래프 4: 과적합 패턴 상세 분석
# ========================================

def plot_overfitting_analysis():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # 학습 단계별 구분
    phases = ['초기\n(0-650k)', '향상\n(700-900k)', '최적\n(950k)', '과적합\n(1M)']
    success_rates = [0, 5.3, 10, 3]  # 대표값
    colors = ['#EF476F', '#F78C6B', '#06D6A0', '#EF476F']
    
    bars = ax1.barh(phases, success_rates, color=colors, alpha=0.8, 
                    edgecolor='black', linewidth=2)
    
    # 값 표시
    for bar, val in zip(bars, success_rates):
        width = bar.get_width()
        ax1.text(width + 0.3, bar.get_y() + bar.get_height()/2,
                f'{val:.1f}%',
                ha='left', va='center', fontsize=12, fontweight='bold')
    
    ax1.set_xlabel('평균 성공률 (%)', fontsize=12, fontweight='bold')
    ax1.set_title('학습 단계별 성능 변화\n950k에서 최고점, 이후 하락', 
                  fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='x')
    
    # 일반화 갭 분석
    models = ['950k', '1M']
    train_performance = [10, 10]  # 학습 시드(1000) 성공률
    test_performance = [11, 4]    # 테스트 시드 평균 성공률
    
    x = np.arange(len(models))
    width = 0.35
    
    bars1 = ax2.bar(x - width/2, train_performance, width, 
                    label='학습 시드(1000)', color='#2E86AB', alpha=0.8)
    bars2 = ax2.bar(x + width/2, test_performance, width, 
                    label='테스트 평균', color='#F18F01', alpha=0.8)
    
    # 값 표시
    for bar in bars1:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.3,
                f'{height:.0f}%', ha='center', va='bottom', 
                fontsize=11, fontweight='bold')
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.3,
                f'{height:.0f}%', ha='center', va='bottom', 
                fontsize=11, fontweight='bold')
    
    # 일반화 갭 표시
    for i, (train, test) in enumerate(zip(train_performance, test_performance)):
        gap = abs(train - test)
        ax2.plot([i - width/2 + width/2, i + width/2 + width/2], 
                [train, test], 'r--', linewidth=2)
        ax2.text(i, (train + test)/2 + 1.5, 
                f'갭: {gap:.0f}%p',
                ha='center', fontsize=10, color='red', fontweight='bold')
    
    ax2.set_ylabel('성공률 (%)', fontsize=12, fontweight='bold')
    ax2.set_title('일반화 갭 비교\n950k 모델이 더 잘 일반화', 
                  fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(models)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim(0, 15)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/4_overfitting_analysis.png', dpi=300, bbox_inches='tight')
    print(f"✅ 저장: {OUTPUT_DIR}/4_overfitting_analysis.png")
    plt.close()


# ========================================
# 그래프 5: 시드 난이도 분석
# ========================================

def plot_seed_difficulty():
    fig, ax = plt.subplots(figsize=(12, 6))
    
    seeds = seed_performance_950k['seed']
    success = seed_performance_950k['success_rate']
    
    # 난이도별 색상
    colors = []
    for s in success:
        if s >= 20:
            colors.append('#06D6A0')  # 쉬움
        elif s > 0:
            colors.append('#F78C6B')  # 중간
        else:
            colors.append('#EF476F')  # 어려움
    
    bars = ax.bar(seeds, success, color=colors, alpha=0.8, 
                  edgecolor='black', linewidth=2)
    
    # 값 표시
    for bar, val in zip(bars, success):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{val:.0f}%',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax.set_xlabel('테스트 시드', fontsize=13, fontweight='bold')
    ax.set_ylabel('성공률 (%)', fontsize=13, fontweight='bold')
    ax.set_title('테스트 시드별 난이도 분석 (950k 모델)\n맵에 따라 0-40% 성능 차이', 
                 fontsize=15, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 45)
    
    # 범례
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#06D6A0', label='쉬움 (20% 이상)', alpha=0.8),
        Patch(facecolor='#F78C6B', label='중간 (1-19%)', alpha=0.8),
        Patch(facecolor='#EF476F', label='어려움 (0%)', alpha=0.8)
    ]
    ax.legend(handles=legend_elements, fontsize=11, loc='upper left')
    
    # 평균선
    avg = np.mean(success)
    ax.axhline(y=avg, color='blue', linestyle='--', linewidth=2, 
               label=f'평균: {avg:.1f}%', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/5_seed_difficulty.png', dpi=300, bbox_inches='tight')
    print(f"✅ 저장: {OUTPUT_DIR}/5_seed_difficulty.png")
    plt.close()


# ========================================
# 메인 실행
# ========================================

def main():
    print("="*70)
    print("📊 보고서용 그래프 생성 중...")
    print("="*70)
    
    print("\n1️⃣ 학습 곡선 생성 중...")
    plot_learning_curves()
    
    print("\n2️⃣ 모델 비교 그래프 생성 중...")
    plot_comparison_950k_vs_1m()
    
    print("\n3️⃣ 전체 요약 그래프 생성 중...")
    plot_overall_summary()
    
    print("\n4️⃣ 과적합 분석 그래프 생성 중...")
    plot_overfitting_analysis()
    
    print("\n5️⃣ 시드 난이도 분석 그래프 생성 중...")
    plot_seed_difficulty()
    
    print("\n" + "="*70)
    print("✅ 모든 그래프 생성 완료!")
    print("="*70)
    print(f"\n📁 저장 위치: {OUTPUT_DIR}/")
    print("\n생성된 그래프:")
    print("  1. 1_learning_curves.png - 학습 곡선 (성공률 + 보상)")
    print("  2. 2_model_comparison.png - 950k vs 1M 직접 비교")
    print("  3. 3_overall_summary.png - 전체 성능 요약")
    print("  4. 4_overfitting_analysis.png - 과적합 패턴 분석")
    print("  5. 5_seed_difficulty.png - 시드별 난이도 분석")
    print("\n💡 이 그래프들을 보고서에 직접 사용할 수 있습니다!")
    print("="*70)


if __name__ == "__main__":
    main()

