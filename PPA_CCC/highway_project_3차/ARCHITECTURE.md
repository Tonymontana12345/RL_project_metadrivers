# 프로젝트 아키텍처 (Modified)

## 📐 전체 시스템 아키텍처

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Configuration Layer                          │
├─────────────────────────────────────────────────────────────────────┤
│  config.py                    │ 기본 설정 (환경, PPO, SAC, TD3)      │
│  config_curriculum.py         │ 커리큘럼 학습 설정                    │
│  config_ppo_curriculum_v*.py  │ PPO 커리큘럼 버전별 설정              │
└─────────────────────────────────────────────────────────────────────┘
                               ↓
┌─────────────────────────────────────────────────────────────────────┐
│                         Core Components                              │
├──────────────────────┬──────────────────────┬──────────────────────┤
│   Environment        │      Agent           │     Utilities        │
├──────────────────────┼──────────────────────┼──────────────────────┤
│ envs/                │ agents/              │ utils/               │
│  └─metadrive_env.py  │  └─rl_agent.py       │  └─path_utils.py     │
│                      │                      │                      │
│ MetaDrive 환경 래퍼  │ PPO/SAC/TD3 팩토리   │ 경로 관리            │
│ - 시드 관리          │ - 모델 생성          │ - 절대경로 변환      │
│ - 환경 설정          │ - 알고리즘 선택      │                      │
└──────────────────────┴──────────────────────┴──────────────────────┘
                               ↓
┌─────────────────────────────────────────────────────────────────────┐
│                         Training Layer                               │
├──────────────────────┬──────────────────────┬──────────────────────┤
│  기본 학습           │  커리큘럼 학습       │  보조 스크립트       │
├──────────────────────┼──────────────────────┼──────────────────────┤
│ train.py             │ train_curriculum.py  │ quick_start.py       │
│  - 단일/다중 시드    │ train_ppo_curriculum │ test_stage1.py       │
│  - PPO/SAC/TD3       │ _v2.py               │ analyze_training     │
│  - 실험 프로토콜     │ train_ppo_curriculum │ _seeds.py            │
│                      │ _v3.py               │ generate_random      │
│                      │                      │ _seeds.py            │
│                      │  - 단계별 학습       │ sample_seeds.py      │
│                      │  - 난이도 조절       │                      │
└──────────────────────┴──────────────────────┴──────────────────────┘
                               ↓
┌─────────────────────────────────────────────────────────────────────┐
│                         Storage Layer                                │
├──────────────────────┬──────────────────────┬──────────────────────┤
│    Models            │      Logs            │     Results          │
├──────────────────────┼──────────────────────┼──────────────────────┤
│ models/              │ logs/                │ results/             │
│  - *.zip (체크포인트)│  - TensorBoard       │  - *.json            │
│  - PPO 모델          │  - 학습 메트릭       │  - 평가 결과         │
│  - SAC 모델          │  - 보상 곡선         │                      │
│  - TD3 모델          │                      │                      │
└──────────────────────┴──────────────────────┴──────────────────────┘
                               ↓
┌─────────────────────────────────────────────────────────────────────┐
│                     Evaluation & Recording Layer                     │
├──────────────────────┬─────────────────────────────────────────────┤
│   모델 평가          │   주행 기록 (Visual Verification)           │
├──────────────────────┼─────────────────────────────────────────────┤
│ evaluate.py          │ Drive_Record.py                             │
│  - 모델 평가         │  - 주행 영상 GIF 녹화                       │
│  - 다중 시드 테스트  │  - 시드별 주행 비교                         │
│  - 성공률 측정       │  - 성공/실패 사례 시각화                    │
│                      │  - Top-down View 렌더링                     │
└──────────────────────┴─────────────────────────────────────────────┘
                               ↓
┌─────────────────────────────────────────────────────────────────────┐
│                        Output (GIFs)                                 │
├─────────────────────────────────────────────────────────────────────┤
│ gifs/                                                                │
│  - driving.gif                                                       │
│  - seed_*.gif                                                        │
│  - comparison/                                                       │
│     └─ success.gif, failure.gif                                      │
└─────────────────────────────────────────────────────────────────────┘
```

## 🔄 데이터 흐름

```
1. Configuration
   ├─ config.py → 환경 & 하이퍼파라미터 설정
   └─ config_curriculum_v*.py → 커리큘럼 학습 설정
                ↓
2. Environment Creation
   └─ envs/metadrive_env.py → MetaDrive 환경 생성 (시드 기반)
                ↓
3. Agent Creation
   └─ agents/rl_agent.py → PPO/SAC/TD3 모델 생성
                ↓
4. Training
   ├─ train.py → 기본 학습 (단일/다중 시드)
   └─ train_ppo_curriculum_v*.py → 커리큘럼 학습 (단계별)
                ↓
5. Model Checkpoints
   └─ models/*.zip → 체크포인트 저장
                ↓
6. Evaluation & Recording
   ├─ evaluate.py → 정량적 평가 (수치 데이터)
   └─ Drive_Record.py → 정성적 평가 (GIF 영상)
                ↓
7. Final Output
   ├─ results/*.json → 평가 결과 데이터
   └─ gifs/*.gif → 주행 영상 기록
```

## 📦 주요 모듈 상세

### 1. 환경 모듈 (envs/)
```
metadrive_env.py
├─ make_env(seed, render, config)
│   ├─ MetaDriveEnv 인스턴스 생성
│   ├─ 시드 설정
│   └─ 환경 설정 적용
└─ 반환: gymnasium.Env
```

### 2. 학습 모듈 (Training)
```
train_ppo_curriculum_v3.py (커리큘럼 학습)
├─ 단계별 환경 생성
│   ├─ Stage 1: 간단한 맵
│   ├─ Stage 2: 중간 난이도
│   └─ Stage 3: 복잡한 맵
├─ 전이 학습 (이전 단계 모델 로드)
└─ 단계별 모델 저장
```

### 3. 기록 및 시각화 모듈 (Recording)
**`Drive_Record.py`** (핵심 시각화 도구)
이 모듈은 `visualize.py` 등의 별도 시각화 도구 대신 사용되며, 실제 주행 영상을 기록하여 모델의 행동을 직접 검증합니다.

```python
record_driving_gif(...)
├─ 모델 로드 (PPO/SAC/TD3 자동 감지)
├─ MetaDrive 환경 생성 (Render Mode: Top-down)
├─ 에피소드 실행 (model.predict)
├─ 프레임 캡처 및 GIF 생성
└─ 결과 저장 (gifs/)

주요 기능:
- 단일 에피소드 녹화
- 다중 시드 비교 녹화
- 성공/실패 케이스 비교 녹화
```

## 🎯 실험 워크플로우

### 워크플로우 1: 학습 및 영상 검증
```
1. 학습 실행
   python train_ppo_curriculum_v3.py
   ↓
2. 모델 생성
   models/ppo_5_stage_curriculum_final.zip
   ↓
3. 주행 영상 녹화 (검증)
   python Drive_Record.py --model models/ppo_5_stage_curriculum_final.zip --seed 2000
   ↓
4. 결과 확인
   gifs/driving.gif 확인
```

### 워크플로우 2: 성공/실패 사례 분석
```
1. 평가 실행 (성공/실패 시드 식별)
   python evaluate.py ...
   ↓
2. 비교 영상 생성
   python Drive_Record.py --model ... --success 1000 --failure 2679
   ↓
3. 원인 분석
   gifs/comparison/success.gif vs failure.gif 비교
```

---

**이 아키텍처는 `Drive_Record.py`를 통한 직접적인 시각적 검증(Visual Verification)을 중심으로 구성되어 있습니다.** 🚗🎥
