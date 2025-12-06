# MetaDrive 자율주행 강화학습 프로젝트 (TD3_JSK)

이 프로젝트는 **MetaDrive** 시뮬레이션 환경에서 **TD3 (Twin Delayed DDPG)** 강화학습 알고리즘을 사용하여, 점진적으로 자율주행 에이전트의 성능과 일반화(Generalization) 능력을 향상시키는 과정을 담고 있습니다.

## 📚 프로젝트 개요

프로젝트는 총 3단계로 구성되어 있으며, 각 단계는 이전 단계의 한계를 극복하고 더 복잡하고 다양한 환경에 적응하도록 설계되었습니다.

| 단계 | 목표 | 핵심 특징 | 학습 환경 |
| :-- | :-- | :-- | :-- |
| **1차** | **Baseline 구축** | 기본 주행 능력 학습 | 고정된 단일 맵 (Seed 1000) |
| **2차** | **Generalization** | 다양한 도로/트래픽 적응 | **TSCO** 복합 맵 + 다중 시드 (6개) |
| **3차** | **Domain Randomization** | 미지의 맵 적응력(Robustness) | **Random Block Order** (매 에피소드 맵 변경) |

---

## 📂 단계별 학습 가이드

각 폴더(`1차/`, `2차/`, `3차/`) 내부에 해당 단계에 맞는 코드와 상세한 `README.md`가 포함되어 있습니다.

### [1차] 기본 주행 학습 (Baseline)
*   **목표**: 고정된 환경에서 트랙을 이탈하지 않고 목적지까지 도달하는 기본적인 주행 정책 학습.
*   **실행**:
    ```bash
    cd 1차
    python train.py --mode fixed --algorithm td3
    ```

### [2차] 일반화 성능 향상 (Generalization)
*   **목표**: 교차로(T), 직선(S), 곡선(C), 로터리(O)가 결합된 복잡한 **TSCO** 맵 구조 학습 및 다중 시드 훈련을 통한 일반화.
*   **실행**:
    ```bash
    cd 2차
    python train.py --mode multi --algorithm td3  # 다중 시드 학습
    ```

### [3차] 도메인 무작위화 (Domain Randomization)
*   **목표**: 에피소드마다 도로 블록(`T, S, C, O`)의 순서가 무작위로 바뀌는 환경에서 학습하여, 맵의 패턴을 외우지 않고 상황을 판단하는 능력 배양.
*   **실행**:
    ```bash
    cd 3차
    python train.py --mode random_blocks --algorithm td3
    ```

---

## 📊 평가 및 시각화 (Evaluation)

학습된 모델의 성능을 검증하고 주행을 시각화하는 방법입니다. 각 차수 폴더로 이동하여 실행하세요.

### 1. 기본 주행 성능 평가 (`evaluate.py`)
지정된 테스트 시드에서 성공률, 충돌률, 평균 보상 등을 측정합니다.

```bash
# 기본 평가 (렌더링 없음, 빠른 속도)
python evaluate.py --model models/your_model_name.zip

# 주행 화면 시각화 (렌더링 켜기)
python evaluate.py --model models/your_model_name.zip --render
```

### 2. 랜덤 맵 조합 평가 (`evaluate_random_maps.py`)
**2차, 3차 프로젝트**에서 사용 가능합니다. 학습 때 보지 못한 새로운 맵 조합(`OTCS`, `SCTO` 등)을 무작위로 생성하여 모델의 진정한 일반화 성능을 테스트합니다.

```bash
# 10개의 새로운 랜덤 맵에서 테스트
python evaluate_random_maps.py --model models/your_model_name.zip --num-maps 10
```

### 3. 주행 영상 녹화 (`Drive_Record.py`)
**3차 프로젝트**에 포함된 기능을 통해 주행 과정을 GIF로 저장할 수 있습니다. 

```bash
cd 3차
# 단일 에피소드 녹화 (driving.gif 저장)
python Drive_Record.py --model models/your_model_name.zip --seed 1000

# 여러 에피소드 녹화 (gifs/ 폴더에 저장)
python Drive_Record.py --model models/your_model_name.zip --seed 1000 --episodes 3

# 여러 시드 일괄 녹화 (각 시드별로 GIF 생성)
python Drive_Record.py --model models/your_model_name.zip --seeds 1000 2000 3000
```

### 📈 결과물 확인
평가가 완료되면 `results/` 폴더에 다음 파일들이 생성됩니다.
*   **evaluation_results.json**: 평가 수치 데이터
*   **evaluation_report.txt**: 요약 리포트
*   **figures/*.png**: 학습 곡선 및 맵별 성능 비교 그래프

---

## 🛠️ 설치 및 요구사항 (Requirements)

이 프로젝트를 실행하기 위해 다음 패키지들이 필요합니다.

```bash
pip install metadrive-simulator stable-baselines3 torch numpy pandas matplotlib seaborn tqdm
```

*   **Python**: 3.8 이상 권장
*   **MetaDrive**: 0.4.x 이상
*   **Stable-Baselines3**: 2.0.0 이상
