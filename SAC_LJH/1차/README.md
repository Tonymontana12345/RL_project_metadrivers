# MetaDrive 자율주행 강화학습 프로젝트 (PPO)

MetaDrive 시뮬레이터를 활용한 자율주행 에이전트 학습 및 일반화 성능 평가 프로젝트입니다.

## 📋 프로젝트 목표

1.  **고정 시드 학습**: 단일 맵(Seed 1000)에서 **PPO 알고리즘**으로 주행 정책을 완벽하게 학습합니다.
2.  **일반화 성능 검증**: 학습 과정에서 보지 못한 새로운 맵(Seed 2000~2004)에서 에이전트가 얼마나 잘 주행하는지 평가합니다.

---

## 🚀 PPO 학습 및 실행 가이드

이 프로젝트는 **PPO(Proximal Policy Optimization)** 알고리즘을 메인으로 사용합니다.

### 1. 환경 준비
```bash
# 가상환경 활성화
source venv_pgdrive/bin/activate
```

### 2. 학습 시작 (Training)
PPO 알고리즘을 사용하여 고정 시드 환경에서 학습을 진행합니다.
```bash
# PPO 학습 실행
python train.py --mode fixed --algorithm ppo
```
*   **과정**: 설정된 10만 스텝 동안 데이터를 수집하고 모델을 업데이트합니다.
*   **저장**: 학습된 모델은 `models/ppo_fixed_seed_1000.zip`으로 저장됩니다.
*   **설정**: 학습률 등 상세 설정은 `config.py`의 `PPO_CONFIG`에서 변경 가능합니다.

### 3. 평가 및 시각화 (Evaluation)
학습된 모델이 실제로 어떻게 주행하는지 눈으로 확인하고 성능을 측정합니다.

**👁️ 주행 화면 보기 (Rendering)**
```bash
# 시각화 모드로 실행 (직접 주행 화면 확인)
python evaluate.py --model models/ppo_fixed_seed_1000.zip --render
```

**📊 성능 측정 (Benchmark)**
```bash
# 20번의 에피소드 동안 성능 측정 (성공률, 보상 등)
python evaluate.py --model models/ppo_fixed_seed_1000.zip --episodes 20
```

---

## � 실험 내용

| 구분 | 설명 | 시드(Seed) | 목표 |
| :-- | :-- | :-- | :-- |
| **학습 (Train)** | 에이전트 훈련 | `1000` (고정) | 트랙 완주 및 충돌 회피 학습 |
| **평가 (Eval)** | 일반화 성능 테스트 | `2000` ~ `2004` | 새로운 도로 환경 적응력 검증 |

**평가 지표:**
*   **Success Rate**: 목적지 도착 확률
*   **Generalization Gap**: (학습 환경 점수) - (테스트 환경 점수) 차이

---

## � 프로젝트 구조

```
highway_project_1차/
├── train.py            # 학습 실행 스크립트
├── evaluate.py         # 모델 평가 및 시각화 스크립트
├── config.py           # 전체 설정 (환경, PPO 파라미터 등)
├── agents/             # RL 에이전트 정의 폴더
├── models/             # 학습된 모델 저장 폴더
└── results/            # 평가 결과 저장 폴더
```
