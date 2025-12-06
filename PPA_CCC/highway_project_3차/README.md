# MetaDrive 자율주행 강화학습 프로젝트 (PPO) - 3차 완성형

MetaDrive 시뮬레이터의 **최종 완성 버전(v3)**입니다. 초보 운전자처럼 기초부터 심화까지 단계적으로 배우는 **8단계 커리큘럼 학습**과 **기억력(LSTM)**을 갖춘 PPO 모델을 사용합니다.

## � 프로젝트 목표 (v3 핵심)

1.  **8단계 정밀 커리큘럼**: 직선 주행부터 장애물 회피, 복잡한 도심 주행까지 물 흐르듯 자연스럽게 난이도를 높여 학습 실패를 방지합니다.
2.  **칭찬형 보상 설계**: 초기에 과도한 페널티를 없애고 전진만 해도 큰 보상을 주어, 에이전트가 겁먹지 않고 빠르게 적응하도록 유도합니다.
3.  **Recurrent PPO (LSTM)**: 과거의 주행 상황을 기억하는 신경망을 사용하여 순간적인 판단이 아닌 맥락을 고려한 주행을 합니다.

---

## 🚀 PPO 학습 및 실행 가이드

이 프로젝트는 **PPO(Proximal Policy Optimization)** 알고리즘에 특화되어 있습니다.

### 1. 환경 준비
```bash
# 가상환경 활성화
source venv_pgdrive/bin/activate
```

### 2. 학습 시작 (Training) - 강력 추천 ⭐

가장 성능이 뛰어난 **v3 커리큘럼 학습**을 시작합니다.
```bash
# 8단계 커리큘럼 학습 실행
python train_ppo_curriculum_v3.py
```
*   **특징**: Stage 1(직선) → Stage 8(복잡한 도심)까지 자동으로 진행됩니다.
*   **소요 시간**: 전체 약 200만 스텝 (수 시간 소요)
*   **저장**: 각 스테이지가 끝날 때마다 모델이 `models/` 폴더에 저장됩니다.

### 3. 평가 및 시각화 (Evaluation)

학습된 모델이 실제로 어떻게 주행하는지 확인합니다.

**📊 최종 모델 평가**
```bash
# 최종 학습된 모델(v3) 평가
python evaluate.py --model models/ppo_8_stage_curriculum_v3_final.zip --episodes 20
```

**👁️ 주행 화면 보기 (Rendering)**
```bash
# 시각화 모드로 주행 실력 확인
python evaluate.py --model models/ppo_8_stage_curriculum_v3_final.zip --render
```

**🎞️ 주행 영상 기록**
```bash
# 주행 영상을 GIF로 저장
python Drive_Record.py
```

---

## � 8단계 커리큘럼 구성

| 단계 | 이름 | 목표 및 특징 |
| :-- | :-- | :-- |
| **Stage 1** | Learn to Move | 직선 도로. 전진 보상 극대화. "일단 가보자!" |
| **Stage 2** | Master Straight | 직선 도로 마스터. 차선 유지 집중. |
| **Stage 3** | Single Curve | 완만한 커브 하나를 도는 법 학습. |
| **Stage 4** | Transition | 직선에서 커브로 진입하는 감각 익히기. |
| **Stage 5** | Complex Road | 교통량 없는 복잡한 도로(TSCO) 완주. |
| **Stage 6** | Obstacle | 정지된 장애물 차량 1대 회피. |
| **Stage 7** | Light Traffic | 적은 수의 움직이는 차량들 사이로 주행. |
| **Stage 8** | **Final Challenge** | **복잡한 맵 + 많은 교통량**. 최종 관문. |

---

## � 프로젝트 구조

```
highway_project_3차/
├── config_ppo_curriculum_v3.py  # [핵심] 8단계 커리큘럼 및 보상 설정
├── train_ppo_curriculum_v3.py   # [핵심] 커리큘럼 학습 실행 스크립트
├── evaluate.py                  # 모델 평가 스크립트
├── Drive_Record.py              # 주행 영상 녹화
├── config.py                    # 기본 설정
└── models/                      # 학습된 모델 저장소
```
