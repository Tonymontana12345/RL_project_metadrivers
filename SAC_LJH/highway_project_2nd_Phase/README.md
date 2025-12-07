# MetaDrive 자율주행 강화학습 프로젝트 (2nd Phase)

SAC 알고리즘 기반 자율주행 강화학습 실험 프로젝트 - 커리큘럼 학습 적용

## 📋 프로젝트 개요

### 목표
- **커리큘럼 학습**: 단계적 난이도 증가로 효율적인 학습
- **SAC 알고리즘 최적화**: Soft Actor-Critic 알고리즘 기반 자율주행
- **일반화 성능 평가**: 다양한 시드와 맵 환경에서 성능 측정
- **다양한 맵 환경**: Straight, T-intersection, Circular, Roundabout

### 핵심 특징
- ✅ **MetaDrive**: 절차적 생성 기반 자율주행 시뮬레이터
- ✅ **커리큘럼 학습**: 6단계 점진적 난이도 증가 시스템
- ✅ **SAC 알고리즘**: Off-policy로 샘플 효율적 학습 (Stable-Baselines3)
- ✅ **다중 알고리즘 지원**: PPO, SAC, TD3
- ✅ **시각화 도구**: 궤적, 맵, 주행 영상 녹화 기능
- ✅ **재현성**: 고정 시드로 동일한 환경 보장

---

## 🚀 사용 방법

### 1. 환경 테스트

```bash
# 환경 데모
python quick_start.py --demo

# 수동 제어
python quick_start.py --manual
```

### 2. 학습

#### 기본 학습
```bash
# SAC 알고리즘으로 학습 (권장)
python train.py --mode fixed --algorithm sac

# PPO 알고리즘
python train.py --mode fixed --algorithm ppo

# TD3 알고리즘
python train.py --mode fixed --algorithm td3

# 다중 시드 학습
python train.py --mode multi --algorithm sac
```

#### 커리큘럼 학습 (2nd Phase 신규)
```bash
# Stage 3부터 학습
python train_curriculum.py —start_stage 3 —end_stage 6

# Stage 6만 재학습
python train_curriculum.py —start_stage 6 —end_stage 6

# 커스텀 총 타임스텝 설정
python train_curriculum.py --algorithm sac --total-timesteps 3000000
```

### 3. 평가

#### 기본 평가
```bash
# 모델 평가 (알고리즘 자동 감지)
python evaluate.py --model models/SAC/sac_stage6_ramps.zip

# 에피소드 수 지정
python evaluate.py --model models/SAC/sac_stage6_ramps.zip --episodes 50

# 렌더링과 함께
python evaluate.py --model models/SAC/sac_stage6_ramps.zip --render
```



### 4. 시각화

```bash
# 평가 결과 시각화
python visualize.py --results results/evaluation_results.json

# 학습 모니터링
tensorboard --logdir logs/
```

---

## 🔍 시각화 도구

### 1. 차량 궤적 시각화 (실패 지점 확인)
```bash
# 실패가 많은 시드 2679 분석
python visualize_trajectory.py --model models/SAC/sac_stage6_ramps.zip --seed 2679 --episodes 5

# 여러 시드 비교
python visualize_trajectory.py --model models/SAC/sac_stage6_ramps.zip --seeds 2679 3286 4657 --compare
```

### 2. 맵 시각화 (Top-down View)
```bash
# 모든 시드 맵 비교
python topdown.py --save

# 특정 시드만
python topdown.py --seeds 1000 2679 --save
```

### 3. 주행 영상 녹화 (GIF)

#### 기본 녹화
```bash
# 단일 에피소드 녹화
python Drive_Record.py --model models/SAC/sac_stage6_ramps.zip --seed 1000

# 커스텀 파일명
python Drive_Record.py --model models/SAC/sac_stage6_ramps.zip --seed 1000 --output my_driving.gif
```

#### 여러 에피소드 녹화
```bash
# 여러 에피소드
python Drive_Record.py --model models/SAC/sac_stage6_ramps.zip --seed 2679 --episodes 3
```

#### 비교 분석
```bash
# 여러 시드 비교
python Drive_Record.py --model models/SAC/sac_stage6_ramps.zip --seeds 1000 2679 4657 --compare

# 성공 vs 실패 비교
python Drive_Record.py --model models/SAC/sac_stage6_ramps.zip --success 1000 --failure 2679
```

---


## 📁 프로젝트 구조

```
highway_project_2nd_Phase/
├── config.py                      # 설정 (시드, 하이퍼파라미터, 커리큘럼)
├── train.py                       # 기본 학습 스크립트
├── train_curriculum.py            # 커리큘럼 학습 스크립트 (신규)
├── evaluate.py                    # 기본 평가 스크립트
├── evaluate_curriculum.py         # 커리큘럼 평가 스크립트 (신규)
├── visualize.py                   # 결과 시각화
├── visualize_trajectory.py        # 궤적 시각화
├── Drive_Record.py                # 주행 영상 녹화 (GIF)
├── topdown.py                     # 맵 시각화 (Top-down View)
├── quick_start.py                 # 환경 테스트
│
├── agents/
│   └── rl_agent.py               # RL 에이전트 (PPO, SAC, TD3)
│
├── envs/
│   └── metadrive_env.py          # MetaDrive 환경 래퍼
│
├── utils/
│   └── path_utils.py             # 경로 유틸리티
│
├── models/
│   └── SAC/                      # SAC 모델 저장
│       ├── curriculum/           # 커리큘럼 학습 모델 (단계별)
│       └── sac_stage6_ramps.zip  # 최종 모델
│
├── logs/                          # TensorBoard 로그
├── results/                       # 평가 결과
├── gifs/                          # 주행 영상 GIF
│
├── CURRICULUM_LEARNING_GUIDE.md   # 커리큘럼 학습 가이드
├── Drive_Record.md                # GIF 녹화 가이드
└── Trajectory&Map_TopDown.md      # 시각화 가이드
```

---

## ⚙️ 주요 설정 (config.py)

### 커리큘럼 학습 설정 (신규)

```python
# Stage 1: 로터리만 (로터리 집중 학습!)
STAGE1_ENV_CONFIG = {
    "map": "O",  # rOundabout (로터리만)
    "traffic_density": 0.05,
    "horizon": 1500,
}

# Stage 2: 로터리 + 곡선
STAGE2_ENV_CONFIG = {
    "map": "OC",  # rOundabout + Curve
    "num_scenarios": 5,
    "traffic_density": 0.08,
    "horizon": 1500,
}

# ... (총 6단계)

# Stage 6: 램프 추가 (최종 난이도)
STAGE6_ENV_CONFIG = {
    "map": "TSCOXrR",  # T + Straight + Curve + rOundabout + X + InRamp + OutRamp
    "traffic_density": 0.12,
    "num_scenarios": 50,
    "horizon": 1500,
}
```

### 환경 설정

```python
FIXED_SEED_ENV_CONFIG = {
    "map": "TSCO",              # T-intersection, Straight, Circular, rOundabout
    "traffic_density": 0.1,     # 차량 밀도
    "horizon": 1000,            # 최대 스텝
    "num_scenarios": 10,        # 시나리오 수

    # 센서 설정
    "vehicle_config": {
        "lidar": {"num_lasers": 72, "distance": 50},
    },
}
```

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

### 시드 설정

```python
FIXED_SEED = 1000
TRAIN_SEEDS = [1409, 2824, 5506, 6339, 8576, 4806]  # 학습용 6개
TEST_SEEDS = [2679, 3286, 4657, 5012, 9935]          # 평가용 5개
```

---

## 🎓 커리큘럼 학습 시스템

### 6단계 난이도 증가

| 단계 | 이름 | 맵 | 시나리오 수 | 교통량 | 학습 스텝 | 특징 |
|------|------|-----|---------|--------|-----------|------|
| 1 | roundabout | O | 1 | 0.05 | 150k | 로터리 집중 학습 |
| 2 | roundabout_curve | OC | 5 | 0.08 | 250k | 로터리 + 곡선 |
| 3 | toc | TOC | 10 | 0.08 | 250k | T교차로 + 로터리 + 곡선 |
| 4 | final | TOCS | 20 | 0.1 | 300k | 직선 추가 |
| 5 | intersection | TSCOX | 30 | 0.1 | 300k | 십자교차로 추가 |
| 6 | ramps | TSCOXrR | 50 | 0.12 | 300k | 램프 포함 최종 난이도 |

### 커리큘럼 학습 장점
- 단계적 학습으로 수렴 안정성 향상
- 복잡한 환경에서의 성능 개선
- 학습 초기 실패 감소
- 더 나은 일반화 성능

---

#

## 🔧 트러블슈팅

### 학습이 너무 느림

```python
# config.py 수정
FIXED_SEED_ENV_CONFIG = {
    "use_render": False,        # 렌더링 끄기
    "decision_repeat": 10,      # 5 → 10 (더 빠름)
}
```

### 메모리 부족

```python
# config.py 수정
SAC_CONFIG = {
    "buffer_size": 100000,      # 500k → 100k
    "batch_size": 128,          # 256 → 128
}
```

---
 
---

**Happy Learning! 🚗**
