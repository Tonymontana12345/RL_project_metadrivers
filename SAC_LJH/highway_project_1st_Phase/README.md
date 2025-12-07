# MetaDrive 자율주행 강화학습 프로젝트

SAC 알고리즘 기반 자율주행 강화학습 실험 프로젝트

## 📋 프로젝트 개요

### 목표
- **SAC 알고리즘 학습**: TSCO 맵에서 Soft Actor-Critic 알고리즘 최적화
- **일반화 성능 평가**: 다양한 시드에서 성능 측정
- **TSCO 맵 마스터**: T-intersection, Straight, Circular, rOundabout 블록 조합

### 핵심 특징
- ✅ **MetaDrive**: 절차적 생성 기반 자율주행 시뮬레이터
- ✅ **SAC 알고리즘**: Off-policy로 샘플 효율적 학습 (Stable-Baselines3)
- ✅ **다중 알고리즘 지원**: PPO, SAC, TD3
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

### 3. 평가

```bash
# 모델 평가 (알고리즘 자동 감지)
python evaluate.py --model models/SAC/sac_multi_seed.zip

# 에피소드 수 지정
python evaluate.py --model models/SAC/sac_multi_seed.zip --episodes 50

# 렌더링과 함께
python evaluate.py --model models/SAC/sac_multi_seed.zip --render
```

### 4. 시각화

```bash
# 평가 결과 시각화
python visualize.py --results results/evaluation_results.json

# 학습 모니터링
tensorboard --logdir logs/
```

---

## 이제 사용 가능한 명령어

### 1. 차량 궤적 시각화 (실패 지점 확인)
```bash
# 실패가 많은 시드 2679 분석
python visualize_trajectory.py --model models/SAC/sac_multi_seed.zip --seed 2679 --episodes 5

# 여러 시드 비교
python visualize_trajectory.py --model models/SAC/sac_multi_seed.zip --seeds 2679 3286 4657 --compare
```

### 2. 맵 시각화
```bash
# 모든 시드 맵 비교
python topdown.py --save
```

### 3. 주행 영상 녹화 (GIF)
```bash
# 단일 에피소드 녹화
python Drive_Record.py --model models/SAC/sac_multi_seed.zip --seed 1000

# 커스텀 파일명
python Drive_Record.py --model models/SAC/sac_multi_seed.zip --seed 1000 --output my_driving.gif

# 실패하는 시드 녹화 (디버깅용)
python Drive_Record.py --model models/SAC/sac_multi_seed.zip --seed 2679 --output failure_seed_2679.gif
```


## 📁 프로젝트 구조

```
highway_project_1st_Phase/
├── config.py                      # 설정 (시드, 하이퍼파라미터)
├── train.py                       # 학습 스크립트
├── evaluate.py                    # 평가 스크립트
├── visualize.py                   # 결과 시각화
├── visualize_trajectory.py        # 궤적 시각화
├── Drive_Record.py                # 주행 영상 녹화
├── topdown.py                     # 맵 시각화
├── quick_start.py                 # 환경 테스트
│
├── agents/
│   └── rl_agent.py               # RL 에이전트 (PPO, SAC, TD3)
│
├── envs/
│   └── metadrive_env.py          # MetaDrive 환경 래퍼
│
├── models/
│   └── SAC/                      # SAC 모델 저장
│       ├── sac_multi_seed.zip
│       └── sac_quick_test.zip
│
├── logs/                          # TensorBoard 로그
└── results/                       # 평가 결과
```

---

## ⚙️ 주요 설정 (config.py)

### 환경 설정

```python
FIXED_SEED_ENV_CONFIG = {
    "map": "TSCO",              # T-intersection, Straight, Circular, rOundabout
    "traffic_density": 0.1,     # 차량 밀도
    "horizon": 1000,            # 최대 스텝

    # 센서 설정
    "vehicle_config": {
        "lidar": {"num_lasers": 72, "distance": 50},  # 72개 레이저, 50m
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

## 📊 알고리즘 비교

| 특징 | PPO | SAC | TD3 |
|------|-----|-----|-----|
| **타입** | On-policy | Off-policy | Off-policy |
| **샘플 효율성** | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **안정성** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| **학습 속도** | 느림 | 빠름 | 빠름 |
| **메모리** | 중간 | 많음 | 많음 |

---

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
