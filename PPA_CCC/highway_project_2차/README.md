# MetaDrive 자율주행 강화학습 프로젝트 (PPO) - 2차 심화

MetaDrive 시뮬레이터를 활용한 자율주행 에이전트 학습 프로젝트의 **심화 버전(v2)**입니다. 복잡한 도심 주행을 위한 **TSCO 맵**과 **3단계 커리큘럼 학습**을 사용하여 PPO 에이전트를 훈련시킵니다.

## 📋 프로젝트 목표

1.  **TSCO 맵 정복**: 단순 도로가 아닌 **교차로(T-Intersection, Straight, Circular, Roundabout)**가 포함된 복잡한 도심형 맵을 주행합니다.
2.  **3단계 커리큘럼 학습**: 쉬운 환경(Easy)부터 어려운 환경(Hard)까지 단계적으로 난이도를 높여 학습 효율을 극대화합니다.
3.  **안전 중심 주행**: 충돌과 도로 이탈을 최소화하고, 다양한 교통량(Traffic Density)에 적응하는 것을 목표로 합니다.

---

## 🚀 PPO 학습 및 실행 가이드

이 프로젝트는 **PPO(Proximal Policy Optimization)** 알고리즘에 최적화되어 있습니다.

### 1. 환경 준비
```bash
# 가상환경 활성화
source venv_pgdrive/bin/activate
```

### 2. 학습 시작 (Training) - 강력 추천 ⭐

가장 효과적인 **커리큘럼 학습**을 시작합니다. 난이도가 자동으로 조절됩니다.
```bash
# 3단계 커리큘럼 학습 실행 (총 150만 스텝)
python train_curriculum.py
```
*   **특징**: Easy(차량 없음) → Medium(중간 교통) → Hard(혼잡 교통) 순서로 진행됩니다.
*   **소요 시간**: 약 5시간 (총 150만 스텝)
*   **저장**: 50,000 스텝마다 모델이 `models/` 폴더에 저장됩니다.

### 3. 평가 및 시각화 (Evaluation)

학습된 모델이 얼마나 잘 주행하는지 확인합니다.

**📊 성능 측정 (Benchmark)**
```bash
# 학습된 커리큘럼 모델 평가 (20 에피소드)
python evaluate.py --model models/ppo_curriculum_v6_balanced_final.zip --episodes 20
```

**👁️ 주행 화면 보기 (Rendering)**
```bash
# 시각화 모드로 주행 실력 확인
python evaluate.py --model models/ppo_curriculum_v6_balanced_final.zip --render
```

**🗺️ TSCO 맵 구조 확인**
```bash
# 학습에 사용된 TSCO 맵 생성 및 확인
python create_traco_maps.py
```

---

## 📊 3단계 커리큘럼 구성

| 단계 | 이름 | 목표 및 특징 | 예상 성공률 |
| :-- | :-- | :-- | :-- |
| **Stage 1** | **Easy** | 차량이 거의 없는 도로(밀도 0.05)에서 기본 주행과 차선 유지 학습. | 60-80% |
| **Stage 2** | **Medium** | 적당한 교통량(밀도 0.10)에서 차량 회피 및 차선 변경 기술 습득. | 50-70% |
| **Stage 3** | **Hard** | 혼잡한 도로(밀도 0.15)에서 돌발 상황 대처 및 일반화 능력 완성. | 60-80% |

---

## 📂 프로젝트 구조

```
highway_project_2차/
├── config_curriculum.py      # [핵심] 3단계 커리큘럼 및 PPO 설정
├── train_curriculum.py       # [핵심] 커리큘럼 학습 실행 스크립트
├── create_traco_maps.py      # TSCO 맵 생성 도구
├── evaluate.py               # 모델 평가 스크립트
├── config.py                 # 기본 설정
└── models/                   # 학습된 모델 저장소
```
