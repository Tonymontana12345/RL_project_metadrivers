# MetaDrive 자율주행 강화학습 프로젝트 (SAC_LJH)

SAC 알고리즘 기반 자율주행 강화학습 실험 프로젝트의 1차, 2차, 3차 통합 README입니다. 이 프로젝트는 단계별로 기능이 확장되었으며, 각 단계는 별도의 하위 폴더에 구성되어 있습니다.

## 📁 프로젝트 구조 및 단계별 특징

### 1️⃣ 1차 프로젝트: 기본 SAC 구현 (`highway_project_1st_Phase`)
- **목표**: TSCO 맵에서 Soft Actor-Critic (SAC) 알고리즘 최적화 및 일반화 성능 평가
- **핵심 특징**:
  - 기본 SAC, PPO, TD3 알고리즘 구현
  - T-intersection, Straight, Circular, rOundabout 블록 조합 맵 학습
  - 고정 시드에서의 안정적인 성능 확보

### 2️⃣ 2차 프로젝트: 커리큘럼 학습 (`highway_project_2nd_Phase`)
- **목표**: 단계적 난이도 증가(Curriculum Learning)를 통한 학습 효율성 증대
- **핵심 특징**:
  - 6단계 난이도 증가 시스템 (로터리 → 복합 교차로 → 램프)
  - 커리큘럼 학습 전용 스크립트 (`train_curriculum.py`)
  - 단계별 모델 평가 및 비교

### 3️⃣ 3차 프로젝트: 일반화 및 시각화 (`highway_project_3th_Phase`)
- **목표**: 랜덤 맵 조합을 통한 일반화 성능 검증 및 심층 시각화
- **핵심 특징**:
  - 랜덤 맵 생성 및 평가 시스템 (`evaluate_random_maps.py`)
  - **TRACO**: 주행 궤적 + 맵 오버레이 시각화 도구
  - 다양한 시나리오에서의 강건성(Robustness) 테스트

---

## 🚀 사용 방법 (통합 가이드)

각 단계별 폴더(`highway_project_Xth_Phase`)로 이동하여 실행해야 합니다.

### 1. 환경 테스트 및 기본 학습 (공통)

```bash
# 해당 단계 폴더로 이동
cd highway_project_3th_Phase  # 예시

# 환경 데모
python quick_start.py --demo

# SAC 알고리즘 학습
python train.py --mode fixed --algorithm sac
```

### 2. 커리큘럼 학습 (2차, 3차)

```bash
# Stage 3부터 학습
python train_curriculum.py —start_stage 3 —end_stage 6

# Stage 6만 재학습
python train_curriculum.py —start_stage 6 —end_stage 6
```

### 3. 랜덤 맵 평가 (3차 전용)

```bash
# 랜덤 맵 평가 실행
python evaluate_random_maps.py --model models/sac/sac_stage6_ramps.zip --num-maps 10
```

---

## 📊 시각화 및 분석 도구

### 1. TRACO (Trajectory + Map) 시각화 (3차)
맵 위에 주행 궤적을 오버레이하여 성공/실패 원인을 직관적으로 분석합니다.

```bash
python create_track_maps_random.py \
    --results results/random_maps_evaluation.json \
    --model models/sac/sac_stage6_ramps.zip \
    --save-dir traco_random
```

### 2. 차량 궤적 및 실패 분석
실패가 많은 시드를 분석하여 주행 패턴을 파악합니다.

```bash
python visualize_trajectory.py --model models/SAC/sac_multi_seed.zip --seed 2679
```

### 3. 주행 영상 녹화 (GIF)

```bash
# 단일 에피소드 녹화
python Drive_Record.py --model models/SAC/sac_multi_seed.zip --seed 1000

# (3차) 특정 맵 문자열로 녹화
python Drive_Record_random.py --map TSCO --seed 1000
```

---

## ⚙️ 주요 설정 및 하이퍼파라미터

모든 설정은 각 폴더의 `config.py`에서 관리됩니다.

### SAC 하이퍼파라미터
```python
SAC_CONFIG = {
    "learning_rate": 3e-4,
    "buffer_size": 500000,      # Off-policy 리플레이 버퍼
    "learning_starts": 5000,
    "batch_size": 256,
    "gamma": 0.99,
    "ent_coef": "auto",         # 엔트로피 자동 조절
}
```

### 커리큘럼 단계 (2차, 3차)
| 단계 | 맵 구성 | 특징 |
|------|---------|------|
| 1 | O | 로터리 집중 학습 |
| 2 | OC | 로터리 + 곡선 |
| 3 | TOC | T교차로 포함 |
| 4 | TOCS | 직선 도로 추가 |
| 5 | TSCOX | 십자 교차로 추가 |
| 6 | TSCOXrR | 램프 포함 최종 난이도 |

---

## 🔧 트러블슈팅

- **학습이 너무 느림**: `config.py`에서 `use_render: False` 및 `decision_repeat: 10` 설정 확인
- **메모리 부족**: `buffer_size`를 100,000으로 줄이고 `batch_size`를 128로 감소

---

**Happy Learning! 🚗**
