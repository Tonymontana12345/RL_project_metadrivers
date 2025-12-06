# MetaDrive 자율주행 강화학습 프로젝트 통합 (MetaDrivers)

이 저장소는 **MetaDrive** 시뮬레이터를 활용하여 다양한 강화학습 알고리즘(PPO, TD3)과 학습 전략(Curriculum Learning, Domain Randomization)을 적용한 자율주행 에이전트 연구 프로젝트 모음입니다.

각 프로젝트는 기초적인 주행 능력 확보부터 복잡한 도심 주행, 그리고 미지의 환경에 대한 일반화 능력까지 점진적으로 발전하는 과정을 담고 있습니다.

## 📂 프로젝트 구성

| 프로젝트 (폴더) | 알고리즘 | 핵심 전략 | 주요 목표 |
| :--- | :--- | :--- | :--- |
| **[PPA_CCC](PPA_CCC)** | **PPO / Recurrent PPO** | **Curriculum Learning** | 단계별 난이도 상승을 통한 복잡한 도심(교차로/로터리) 주행 및 안정성 확보 |
| **[TD3_JSK](TD3_JSK)** | **TD3** | **Generalization** | 다중 시드 및 도메인 무작위화를 통한 미지의 맵 적응력(Robustness) 향상 |

---

## 🚀 1. PPA_CCC (PPO 기반 커리큘럼 러닝)

**"점진적 학습을 통해 복잡한 도심 환경을 정복한다."**

PPA_CCC 프로젝트는 8단계의 정밀한 커리큘럼과 메모리 기반 신경망(LSTM)을 도입하여 학습 실패를 방지하고 주행 안정성을 극대화했습니다.

| 단계 | 구분 | 설명 | 핵심 기술 |
| :-- | :-- | :-- | :-- |
| **1차** | **기초 (Basic)** | 고정 시드(Seed 1000)에서 직선/커브 위주의 기초 주행 학습 | PPO (Basic) |
| **2차** | **심화 (Advanced)** | 교차로/로터리가 포함된 **TSCO 맵** 정복을 위한 3단계 커리큘럼 | PPO (Optimized), 3-Stage Curriculum |
| **3차** | **완성 (Final)** | **8단계 정밀 커리큘럼** 및 **Recurrent PPO(LSTM)** 도입으로 안정성 극대화 | **Recurrent PPO**, 칭찬형 보상 설계 |

👉 **상세 내용 확인**: [PPA_CCC/README.md](PPA_CCC/README.md)

---

## 🚀 2. TD3_JSK (TD3 기반 일반화 성능 향상)

**"어떤 도로 상황에서도 적응할 수 있는 에이전트를 만든다."**

TD3_JSK 프로젝트는 결정론적 정책 기울기(TD3) 알고리즘을 사용하여, 에이전트가 학습하지 않은 새로운 맵 구조에서도 유연하게 대처하는 일반화(Generalization) 능력을 키우는 데 집중했습니다.

| 단계 | 구분 | 설명 | 핵심 기술 |
| :-- | :-- | :-- | :-- |
| **1차** | **Baseline** | 고정된 단일 맵에서 기본 주행 능력 확보 (Baseline 구축) | TD3, Single Seed |
| **2차** | **Generalization** | **다중 시드(6개)** 학습을 통해 다양한 TSCO 맵 패턴 습득 | Multi-Seed Training |
| **3차** | **Domain Randomization** | 매 에피소드마다 맵 블록 순서가 바뀌는 환경에서 적응력 배양 | **Random Block Order**, Robustness |

👉 **상세 내용 확인**: [TD3_JSK/README.md](TD3_JSK/README.md)

---

## 🛠️ 공통 환경 설정 및 설치 (Installation)

모든 프로젝트는 공통된 Python 가상환경에서 실행할 수 있습니다.

### 요구 사항
*   Python 3.8 이상 권장
*   MetaDrive 0.4.x 이상
*   Stable-Baselines3 2.0.0 이상

### 설치 명령어
```bash
# 가상환경 생성 및 활성화 (예시)
python -m venv venv_pgdrive
source venv_pgdrive/bin/activate  # Mac/Linux
# venv_pgdrive\Scripts\activate  # Windows

# 필수 패키지 설치
pip install metadrive-simulator stable-baselines3 torch numpy pandas matplotlib seaborn tqdm
```

## 💻 프로젝트별 상세 실행 가이드 (Step-by-Step Guide)

아래 명령어들은 각 프로젝트의 1차부터 3차까지 순서대로 실행하는 방법입니다.

### 1. PPA_CCC 프로젝트 (PPO + Curriculum)

**위치 이동**
```bash
cd PPA_CCC
```

#### [1차] 기초 주행 (Fixed Map)
```bash
# 학습 실행
python train.py --mode fixed --algorithm ppo

# 평가 실행 (모델 경로는 학습 후 생성된 파일명 확인 필요)
python evaluate.py --model models/ppo_fixed_seed_1000.zip --episodes 20
```

#### [2차] 심화 주행 (TSCO Map + 3-Stage Curriculum)
```bash
# 학습 실행 (커리큘럼 적용)
python train_curriculum.py

# 평가 실행
python evaluate.py --model models/ppo_curriculum_v6_balanced_final.zip --episodes 20
```

#### [3차] 완성형 주행 (8-Stage Curriculum + Recurrent PPO)
```bash
# 학습 실행 (8단계 정밀 커리큘럼 + LSTM)
python train_ppo_curriculum_v3.py

# 평가 실행
python evaluate.py --model models/ppo_8_stage_curriculum_v3_final.zip --episodes 20

# 주행 영상 녹화 (GIF 생성)
python Drive_Record.py
```

---

### 2. TD3_JSK 프로젝트 (TD3 + Generalization)

#### [1차] Baseline 구축 (Single Seed)
```bash
cd TD3_JSK/1차

# 학습 실행
python train.py --mode fixed --algorithm td3

# 평가 실행 (모델명을 실제 생성된 파일명으로 변경하세요)
python evaluate.py --model models/best_model.zip
```

#### [2차] 일반화 성능 향상 (Multi-Seed)
```bash
cd ../2차  # TD3_JSK/2차 폴더로 이동

# 학습 실행 (다중 시드)
python train.py --mode multi --algorithm td3

# 랜덤 맵 평가 (10개 맵 테스트)
python evaluate_random_maps.py --model models/best_model.zip --num-maps 10
```

#### [3차] 도메인 무작위화 (Domain Randomization)
```bash
cd ../3차  # TD3_JSK/3차 폴더로 이동

# 학습 실행 (랜덤 블록 순서)
python train.py --mode random_blocks --algorithm td3

# 주행 영상 녹화
python Drive_Record.py --model models/best_model.zip --seed 1000
```

---

## 📊 결과물 예시

각 프로젝트는 학습 후 다음과 같은 결과물을 제공합니다.
*   **Evaluation Reports**: 성공률, 주행 거리 등 정량적 지표 (`evaluation_results.json`)
*   **Visualizations**: 학습 곡선 그래프, 맵별 성능 비교 차트
*   **Replay GIFs**: 에이전트의 주행 영상 녹화 (`Drive_Record.py` 활용)

