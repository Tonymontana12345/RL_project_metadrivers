# MetaDrive 자율주행 강화학습 프로젝트 (3rd Phase)

SAC 알고리즘 기반 자율주행 강화학습 실험 프로젝트 - 랜덤 맵 일반화 평가

## 📋 프로젝트 개요

### 목표
- **커리큘럼 학습**: 단계적 난이도 증가로 효율적인 학습
- **SAC 알고리즘 최적화**: Soft Actor-Critic 알고리즘 기반 자율주행
- **일반화 성능 평가**: 다양한 시드와 랜덤 맵 조합에서 성능 측정
- **다양한 맵 환경**: Straight, T-intersection, Circular, Roundabout 및 랜덤 조합

### 핵심 특징
- ✅ **MetaDrive**: 절차적 생성 기반 자율주행 시뮬레이터
- ✅ **커리큘럼 학습**: 6단계 점진적 난이도 증가 시스템
- ✅ **SAC 알고리즘**: Off-policy로 샘플 효율적 학습 (Stable-Baselines3)
- ✅ **다중 알고리즘 지원**: PPO, SAC, TD3
- ✅ **랜덤 맵 평가**: 맵 블록의 랜덤 조합으로 일반화 성능 측정 (3rd Phase)
- ✅ **TRACO 시각화**: 맵 + 궤적 오버레이 시각화 도구 (3rd Phase)
- ✅ **시각화 도구**: 궤적, 맵, 주행 영상 녹화 기능
- ✅ **재현성**: 고정 시드로 동일한 환경 보장

---

## 🚀 사용 방법




#### 랜덤 맵 평가 (3rd Phase 신규)
```bash
#기본
python evaluate_random_maps.py \
    --model models/sac/sac_stage6_ramps.zip \

# 랜덤 조합 맵 평가 (T, S, C, O 블록의 랜덤 순열)
python evaluate_random_maps.py \
    --model models/sac/sac_stage6_ramps.zip \
    --num-maps 10 \
    --episodes 10

# 특정 블록만 사용
python evaluate_random_maps.py \
    --model models/sac/sac_stage6_ramps.zip \
    --blocks T S C O \
    --num-blocks 4 \
    --num-maps 20

# 결과 저장 경로 지정
python evaluate_random_maps.py \
    --model models/sac/sac_stage6_ramps.zip \
    --num-maps 15 \
    --output results/my_random_evaluation.json
```


---

## 🔍 시각화 도구


### 3. 주행 영상 녹화 (GIF)


#### 랜덤 맵에서 녹화 (3rd Phase 신규)
```bash
# 특정 맵 문자열로 녹화
python Drive_Record_random.py \
    --model models/sac/sac_stage6_ramps.zip \
    --seed 1000 \
    --map TSCO

# 여러 에피소드
python Drive_Record_random.py \
    --model models/sac/sac_stage6_ramps.zip \
    --seed 2679 \
    --episodes 3 \
    --map STCO

# TEST_SEEDS 전체 자동 녹화
python Drive_Record_random.py \
    --model models/sac/sac_stage6_ramps.zip \
    --all-test-seeds \
    --episodes 3 \
    --map OCST
```

### 4. TRACO 맵 시각화 (Trajectory + Map) - 3rd Phase 신규

TRACO는 맵 위에 주행 궤적을 오버레이한 시각화입니다.

#### 랜덤 맵 평가 결과 시각화
```bash
# 모든 맵의 TRACO 이미지 생성
python create_track_maps_random.py \
    --results results/random_maps_evaluation.json \
    --model models/sac/sac_stage6_ramps.zip \
    --save-dir traco_random

# 성공률 상위 3개 맵만
python create_track_maps_random.py \
    --results results/random_maps_evaluation.json \
    --model models/sac/sac_stage6_ramps.zip \
    --top-k 3 \
    --save-dir traco_top3

# 특정 맵(TSCO, STCO)만 생성
python create_track_maps_random.py \
    --results results/random_maps_evaluation.json \
    --model models/sac/sac_stage6_ramps.zip \
    --maps TSCO STCO \
    --save-dir traco_specific

# 에피소드 수 및 해상도 조정
python create_track_maps_random.py \
    --results results/random_maps_evaluation.json \
    --model models/sac/sac_stage6_ramps.zip \
    --episodes 5 \
    --resolution 2048 2048 \
    --save-dir traco_high_res
```

**TRACO 시각화 특징:**
- 🟢 **초록색 선**: 성공한 주행 궤적
- 🔴 **빨간색 선**: 실패한 주행 궤적 (충돌/도로 이탈)
- 🔵 **파란색 원**: 시작점
- ⭐ **초록색 별**: 성공 종료점 (목적지 도착)
- ✖️ **빨간색 X**: 실패 종료점 (사고 지점)

---


## 📁 프로젝트 구조

```
highway_project_3rd_Phase/
├── config.py                      # 설정 (시드, 하이퍼파라미터, 커리큘럼)
├── train.py                       # 기본 학습 스크립트
├── train_curriculum.py            # 커리큘럼 학습 스크립트
├── evaluate.py                    # 기본 평가 스크립트
├── evaluate_curriculum.py         # 커리큘럼 평가 스크립트
├── evaluate_random_maps.py        # 랜덤 맵 평가 스크립트 (3rd Phase)
├── visualize.py                   # 결과 시각화
├── visualize_trajectory.py        # 궤적 시각화
├── Drive_Record.py                # 주행 영상 녹화 (GIF)
├── Drive_Record_random.py         # 랜덤 맵 주행 녹화 (3rd Phase)
├── create_track_maps_random.py    # TRACO 맵 시각화 (3rd Phase)
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
│   └── sac/                      # SAC 모델 저장
│       ├── curriculum/           # 커리큘럼 학습 모델 (단계별)
│       └── sac_stage6_ramps.zip  # 최종 모델
│
├── logs/                          # TensorBoard 로그
├── results/
│   ├── evaluation_results.json        # 기본 평가 결과
│   ├── random_maps_evaluation.json    # 랜덤 맵 평가 결과 (3rd Phase)
│   └── traco_random/                  # TRACO 이미지 (3rd Phase)
│
├── gifs/                          # 주행 영상 GIF
│   └── auto/                      # 자동 녹화 결과 (맵별/시드별)
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
