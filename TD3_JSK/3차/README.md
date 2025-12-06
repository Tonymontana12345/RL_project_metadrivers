# MetaDrive 자율주행 강화학습 프로젝트 3차 (Domain Randomization)

2차 프로젝트의 일반화(Generalization) 목표를 더욱 발전시켜, **도메인 무작위화(Domain Randomization)** 기법을 적용한 심화 학습 프로젝트입니다.

## 📋 프로젝트 목표

**"어떤 맵이 나와도 주행할 수 있는 에이전트 만들기"**

단순히 여러 시드에서 학습하는 것을 넘어, **에피소드마다 도로 블록의 순서를 무작위로 섞는 환경**에서 에이전트를 학습시킵니다. 이를 통해 특정 맵 패턴(예: TSCO 순서)을 외우는 것이 아니라, 도로의 기하학적 특징(교차로, 곡선 등)을 이해하고 반응하도록 만듭니다.

---

## 🚀 학습 및 실행 가이드

TD3 알고리즘을 사용하여 **무작위 블록 맵(Random Block Map)** 환경에서 학습을 진행합니다.

### 1. 학습 시작 (Training)

**랜덤 블록 맵 학습 (추천)** ✨
매 에피소드마다 `T, S, C, O` 블록이 무작위 순서로 배치되는 환경에서 1M 스텝 동안 학습합니다.
```bash
# --mode random_blocks 옵션 사용
python train.py --mode random_blocks --algorithm td3
```

**기존 모드 학습** (비교용)
```bash
# 2차: 고정된 TSCO 맵 학습
python train.py --mode fixed --algorithm td3

# 2차: 다중 시드 학습
python train.py --mode multi --algorithm td3
```

### 2. 평가 (Evaluation)

**랜덤 맵 조합 평가 (Zero-shot Test)**
학습된 모델이 실제로 다양한 맵 조합(`TSCO`, `OTCS`, `SCTO` 등)에서 얼마나 잘 주행하는지 검증합니다.
```bash
python evaluate_random_maps.py --model models/td3_tsco_random_map_500k.zip --num-maps 10
```
*   `--num-maps 10`: 무작위로 생성한 10개의 서로 다른 맵에서 테스트합니다.

### 3. 주행 영상 녹화 (Visualization) ✨

학습된 모델의 주행 장면을 GIF로 저장하여 눈으로 직접 확인합니다.
```bash
# 랜덤 맵 주행 녹화 (driving.gif 생성)
python Drive_Record.py --model models/td3_tsco_random_map_500k.zip --seed 1000

# 여러 에피소드 연속 녹화 (gifs/ 폴더에 저장)
python Drive_Record.py --model models/td3_tsco_random_map_500k.zip --seed 1000 --episodes 3
```

---

## ⚙️ 주요 변경 사항 (vs 2차)

| 항목 | 2차 (Generalization) | 3차 (Domain Randomization) | 비고 |
| :-- | :-- | :-- | :-- |
| **학습 환경** | 고정된 시드 (Fixed Seeds) | **무작위 블록 순서** (Random Blocks) | 매번 맵 구조 변경 |
| **맵 다양성** | 제한적 (TSCO 구조 유지) | **무한함** (4!=24가지 조합 + 랜덤 파라미터) | 진정한 일반화 |
| **설정 이름** | `MULTI_SEED_ENV_CONFIG` | **`RANDOM_BLOCK_ENV_CONFIG`** | `config.py` 참조 |
| **학습 모드** | `--mode multi` | **`--mode random_blocks`** | `train.py` 참조 |

---

## 🔬 실험 내용 및 기대 효과

### Domain Randomization 효과
에이전트는 학습 중에 수많은 도로 조합을 경험하게 됩니다. 예를 들어:
1.  Episode 1: `T-S-C-O` (교차로 시작)
2.  Episode 2: `O-C-S-T` (로터리 시작)
3.  Episode 3: `S-C-T-O` (직선 시작)
...

이러한 다양성은 에이전트가 "다음에 교차로가 나오면 우회전해야지"라고 외우는 것(Overfitting)을 방지하고, **"센서 데이터를 보니 앞이 막혀있네, 피해야지"**라는 식의 대응 능력(Robustness)을 기르게 합니다.

### 평가 결과물
`results/` 폴더에 다음과 같은 상세 분석 자료가 생성됩니다.
*   **맵별 성능 비교 그래프**: 어떤 맵 조합이 어려웠는지 시각적으로 확인.
*   **JSON 데이터**: 수치적 분석을 위한 원본 데이터.
*   **리포트 텍스트**: 전체적인 성공률과 평균 보상 요약.

---

## 📂 프로젝트 구조

```
3차/
├── train.py                # 학습 스크립트 (random_blocks 모드 추가)
├── evaluate_random_maps.py # 랜덤 맵 평가 스크립트
├── Drive_Record.py         # 주행 영상 GIF 녹화 스크립트 (New ✨)
├── config.py               # RANDOM_BLOCK_ENV_CONFIG 설정 추가
├── envs/                   # MetaDrive 환경 설정
├── agents/                 # RL 에이전트
└── results/                # 결과 저장소
    ├── random_maps_*.png   # 맵별 성능 그래프
    └── random_maps_*.json  # 평가 데이터
```

