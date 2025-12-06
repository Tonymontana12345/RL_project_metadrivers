# MetaDrive 자율주행 강화학습 프로젝트 (PPA_CCC)

이 저장소는 MetaDrive 시뮬레이터를 활용하여 자율주행 에이전트를 단계별로 학습시킨 강화학습 프로젝트 모음입니다. 기초적인 주행부터 복잡한 도심 주행까지 점진적으로 발전하는 과정을 담고 있습니다.

## 📂 프로젝트 구성

| 프로젝트 | 단계 | 주요 특징 | 핵심 기술 |
| :--- | :--- | :--- | :--- |
| **[1차 프로젝트](highway_project_1차)** | 기초 (Basic) | 고정 시드 학습, 일반화 성능 기초 테스트 | PPO (Basic), 직선/커브 위주 |
| **[2차 프로젝트](highway_project_2차)** | 심화 (Advanced) | 복잡한 **TSCO 맵**(교차로/로터리), 3단계 커리큘럼 | PPO (Optimized), Curriculum Learning |
| **[3차 프로젝트](highway_project_3차)** | 완성 (Final) | **8단계 정밀 커리큘럼**, 기억력(LSTM) 적용 | **Recurrent PPO**, 칭찬형 보상 설계 |

---

## 🚀 프로젝트별 상세 내용

### 1. highway_project_1차 (기초)
**"강화학습으로 자율주행이 가능한가?"**를 검증하는 초기 단계입니다.
-   **목표**: 단일 맵(Seed 1000)에서 주행 정책 학습.
-   **환경**: 단순한 5블록 도로.
-   **학습**: `python train.py --mode fixed --algorithm ppo`
-   **평가**: `python evaluate.py --model models/ppo_fixed_seed_1000.zip --episodes 20`

### 2. highway_project_2차 (심화)
**"복잡한 도심에서도 주행할 수 있는가?"**를 목표로 합니다.
-   **목표**: 신호등이 있는 교차로(Intersection)와 로터리(Roundabout)가 포함된 **TSCO 맵** 정복.
-   **방법**: 쉬운 도로에서 어려운 도로로 난이도를 높이는 **3단계 커리큘럼** 도입.
-   **학습**: `python train_curriculum.py`
-   **평가**: `python evaluate.py --model models/ppo_curriculum_v6_balanced_final.zip --episodes 20`
### 3. highway_project_3차 (완성)
**"더 똑똑하고 사람처럼 운전할 수 있는가?"**를 목표로 하는 최종 버전입니다.
-   **목표**: 학습 실패를 방지하고 주행 안정성을 극대화.
-   **방법**:
    1.  **8단계 정밀 커리큘럼**: 아주 기초적인 전진부터 시작해 물 흐르듯 난이도 상향.
    2.  **Recurrent PPO (LSTM)**: 과거 상황을 기억하여 판단하는 신경망 사용.
    3.  **칭찬형 보상**: 초기에 페널티를 줄여 에이전트의 탐험을 독려.
-   **학습**: `python train_ppo_curriculum_v3.py`
-   **평가**: `python evaluate.py --model models/ppo_8_stage_curriculum_v3_final.zip --episodes 20`
-   **GIF 생성**: `python Drive_Record.py` (주행 영상을 GIF로 저장)

---

## 📊 단계별 발전 요약

| 구분 | 1차 프로젝트 | 2차 프로젝트 | 3차 프로젝트 |
| :-- | :-- | :-- | :-- |
| **알고리즘** | PPO (MLP) | PPO (MLP) | **Recurrent PPO (LSTM)** |
| **학습 맵** | Random (5 Blocks) | **TSCO** (Traffic/Straight/Curve/Omni) | **TSCO** (Traffic/Straight/Curve/Omni) |
| **커리큘럼** | 없음 (단일) | 3단계 (Easy → Hard) | **8단계 (Step-by-Step)** |
| **보상 설계** | 기본 | 엄격함 (성공 중심) | **유연함 (초기 격려 → 후기 엄격)** |
| **주요 성과** | 주행 가능성 확인 | 교차로/로터리 주행 성공 | **복잡한 환경 안정성 확보** |

---

## 🛠️ 공통 환경 설정

모든 프로젝트는 동일한 가상환경에서 실행 가능합니다.

```bash
# 가상환경 활성화
source venv_pgdrive/bin/activate

# 패키지 확인
pip list | grep metadrive
```

각 폴더로 이동하여 `README.md`를 참고하면 더 자세한 실행 방법을 확인할 수 있습니다.
