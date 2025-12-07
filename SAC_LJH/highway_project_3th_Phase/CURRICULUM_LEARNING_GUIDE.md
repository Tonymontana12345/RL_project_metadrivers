# 커리큘럼 러닝 가이드

## 📋 프로젝트 개요

**목표**: MetaDrive 환경에서 SAC 알고리즘을 사용해 복잡한 자율주행 환경에서 학습

**핵심 전략**: 로터리 중심 커리큘럼 - 가장 어려운 로터리부터 시작해서 점진적으로 다른 요소 추가

**배경**: 로터리가 가장 어려운 요소이므로, 로터리를 먼저 마스터한 후 다른 요소를 추가하는 전략

---

## 🎯 커리큘럼 러닝 전략 (6단계)

### Stage 1: 로터리만 (O)
- **맵**: "O" (rOundabout만)
- **목표**: 로터리 진입/통과 마스터 (핵심!)
- **학습 시간**: 150k 스텝 (~30분)
- **목표 성공률**: 70%+
- **트래픽 밀도**: 0.05 (매우 낮음 - 로터리 집중 학습)
- **Horizon**: 1500 (로터리 통과 시간)
- **시드**: 단일 시드 (1000)
- **전략**: 가장 어려운 로터리를 먼저 완전히 학습

### Stage 2: 로터리 + 곡선 (OC)
- **맵**: "OC" (rOundabout + Curve)
- **목표**: 로터리 지식 유지 + 곡선 주행 추가
- **학습 시간**: 250k 스텝 (~50분)
- **목표 성공률**: 60%+
- **트래픽 밀도**: 0.08 (낮음)
- **Horizon**: 1500
- **시나리오**: 5개 (다양한 맵 레이아웃)
- **시드**: 단일 시드 (1000)
- **이전 모델**: Stage 1 모델 로드 후 이어서 학습

### Stage 3: T교차로 + 로터리 + 곡선 (TOC)
- **맵**: "TOC" (T-intersection + rOundabout + Curve)
- **목표**: T교차로 추가 (좌회전 학습)
- **학습 시간**: 250k 스텝 (~50분)
- **목표 성공률**: 50%+
- **트래픽 밀도**: 0.08 (낮음)
- **Horizon**: 1500
- **시나리오**: 10개 (복잡도 증가)
- **시드**: 단일 시드 (1000)
- **이전 모델**: Stage 2 모델 로드 후 이어서 학습

### Stage 4: 직선 추가 (TOCS)
- **맵**: "TOCS" (T + rOundabout + Curve + Straight)
- **목표**: 전체 기본 요소 통합
- **학습 시간**: 300k 스텝 (~60분)
- **목표 성공률**: 30%+
- **트래픽 밀도**: 0.1 (중간)
- **Horizon**: 1500
- **시나리오**: 20개 (다양한 조합)
- **시드**: 단일 시드 (1000)
- **이전 모델**: Stage 3 모델 로드 후 이어서 학습

### Stage 5: 십자교차로 추가 (TSCOX)
- **맵**: "TSCOX" (T + Straight + Curve + rOundabout + X-intersection)
- **목표**: 십자교차로 처리 능력 추가
- **학습 시간**: 300k 스텝 (~60분)
- **목표 성공률**: 25%+
- **트래픽 밀도**: 0.1 (중간)
- **Horizon**: 1500
- **시나리오**: 30개 (복잡도 증가)
- **시드**: 단일 시드 (1000)
- **이전 모델**: Stage 4 모델 로드 후 이어서 학습

### Stage 6: 램프 추가 (TSCOXrR)
- **맵**: "TSCOXrR" (T + S + C + O + X + InRamp + OutRamp)
- **목표**: 최종 맵 - 램프 진입/진출 포함
- **학습 시간**: 300k 스텝 (~60분)
- **목표 성공률**: 20%+
- **트래픽 밀도**: 0.12 (높음)
- **Horizon**: 1500
- **시나리오**: 50개 (최대 다양성)
- **시드**: 단일 시드 (1000)
- **이전 모델**: Stage 5 모델 로드 후 이어서 학습

**총 학습 시간**: ~5시간 (1,550k 스텝)

---

## 🔑 핵심 설계 철학

### 1. 로터리 우선 전략
**왜 로터리부터 시작하는가?**
- 로터리는 자율주행에서 가장 복잡한 시나리오 (원형 주행, 진입/진출 타이밍)
- 로터리를 먼저 마스터하면 다른 요소(직선, 곡선, 교차로)는 상대적으로 쉬움
- Bottom-up 접근: 어려운 것 → 쉬운 것 추가

### 2. 점진적 복잡도 증가
**맵 진화 순서**:
```
Stage 1: O (로터리만)
         ↓
Stage 2: OC (+ 곡선)
         ↓
Stage 3: TOC (+ T교차로)
         ↓
Stage 4: TOCS (+ 직선)
         ↓
Stage 5: TSCOX (+ 십자교차로)
         ↓
Stage 6: TSCOXrR (+ 램프)
```

### 3. 시나리오 증가 전략
- **Stage 1-2**: 1-5개 시나리오 (기본 패턴 학습)
- **Stage 3-4**: 10-20개 시나리오 (일반화 시작)
- **Stage 5-6**: 30-50개 시나리오 (강력한 일반화)

### 4. Off-Policy 최적화 (SAC)
- **Replay Buffer 저장/로드**: 이전 Stage 경험 재사용
- **Transfer Learning**: 이전 모델 가중치 로드 후 이어서 학습
- **메모리 효율**: 이전 Buffer 삭제로 디스크 절약

---

## 🔧 주요 설정

### 센서 설정
```python
"vehicle_config": {
    "lidar": {
        "num_lasers": 72,    # 72개 (5° 간격)
        "distance": 70,       # 70m 감지 거리
    },
    "side_detector": {"num_lasers": 2},
    "lane_line_detector": {"num_lasers": 2},
}
```
**발견 사항**: 72개 lidar가 120개보다 성능 우수 (차원의 저주 회피)

### SAC 알고리즘 설정
```python
SAC_CONFIG = {
    "learning_rate": 3e-4,
    "buffer_size": 200000,
    "learning_starts": 5000,     # 5k 스텝 후 학습 시작
    "batch_size": 256,
    "tau": 0.005,
    "gamma": 0.99,               # 장기 보상 중시
    "ent_coef": "auto",          # 자동 엔트로피 조절
}
```

### 보상 함수 (MetaDrive 기본값)
```python
"driving_reward": 1.0,           # 목표 방향 주행
"speed_reward": 0.1,             # 속도 유지
"out_of_road_penalty": 5.0,     # 도로 이탈
"crash_vehicle_penalty": 5.0,   # 차량 충돌
"success_reward": 10.0,         # 목표 도달
```

---

## 💡 주요 학습 내용

### 1. 학습 vs 평가
**Q**: `python train_curriculum.py` 후 개별 모델로 평가하면 되나요?

**A**: 아니요! **각 Stage 모델은 해당 맵에서 평가해야 합니다.**
- Stage 1 모델 → O 맵에서 평가
- Stage 2 모델 → OC 맵에서 평가
- Stage 6 모델 → TSCOXrR 맵에서 평가

**해결책**: `evaluate_curriculum.py` 사용
```bash
python evaluate_curriculum.py                    # 전체 Stage 평가 (1-6)
python evaluate_curriculum.py --stages 1 2 3     # Stage 1-3만 평가
python evaluate_curriculum.py --stages 6         # 최종 Stage만 평가
python evaluate_curriculum.py --n-episodes 20    # 에피소드 수 변경
```

### 2. 성공 기준
**Q**: 평가에서 성공 기준이 뭔가요?

**A**: `info.get("arrive_dest", False)`
- **성공**: 목적지 도착 (맵 끝까지 주행 완료)
- **실패**: 충돌, 도로 이탈, 시간 초과 (horizon)

### 3. Transfer Learning + Replay Buffer
**Q**: Stage 1 모델을 기반으로 Stage 2가 학습되나요?

**A**: 네! SAC의 Off-Policy 특성을 최대한 활용합니다.
```python
# Stage 1: 새 모델
model = SAC("MlpPolicy", env, **SAC_CONFIG)
model.learn(150k steps)
model.save("sac_stage1_roundabout.zip")
model.save_replay_buffer("sac_stage1_roundabout_replay_buffer.pkl")

# Stage 2: Stage 1 로드 + Replay Buffer 로드
model = SAC.load("sac_stage1_roundabout.zip", env=stage2_env)
model.load_replay_buffer("sac_stage1_roundabout_replay_buffer.pkl")
model.learn(250k steps, reset_num_timesteps=False)  # Buffer 유지!
```

이것이 **전이 학습(Transfer Learning) + Off-Policy 최적화**의 핵심입니다.

### 4. Catastrophic Forgetting
**Q**: Stage 2 평가는 좋은데 Stage 1 평가가 나쁘면?

**A**: 정상입니다! 각 모델은 해당 맵에 특화되어 있습니다.
- Stage 2 모델은 OC 맵용입니다
- Stage 1 맵(O)에서 성능 저하는 자연스러움
- **최종 목표는 Stage 6 (TSCOXrR)에서의 성능**

**해결 방법** (필요시):
- 로터리 우선 전략으로 Catastrophic Forgetting 최소화
- Stage 1에서 로터리를 완전히 학습하므로 이후 Stage에서도 유지됨

### 5. 학습 과정 모니터링
**Q**: 학습 과정을 어떻게 보나요?

**A**: 3가지 방법
1. **Progress Bar**: 기본으로 표시됨
2. **TensorBoard**:
   ```bash
   tensorboard --logdir logs
   # http://localhost:6006 접속
   ```
3. **학습 완료 후**:
   ```bash
   python check_training.py
   ```

---

## 📊 예상 결과

### 로터리 우선 전략의 장점
기존 전략(S→SC→ST→TO→SCTO→TSCO)과 달리, 로터리 우선 전략은:
- ✅ 가장 어려운 요소를 먼저 마스터
- ✅ 이후 Stage에서 로터리 지식 유지
- ✅ Catastrophic Forgetting 최소화

### Stage별 예상 성공률
- **Stage 1 (O)**: 70%+ (로터리만 집중 학습)
- **Stage 2 (OC)**: 60%+ (곡선 추가)
- **Stage 3 (TOC)**: 50%+ (T교차로 추가)
- **Stage 4 (TOCS)**: 30%+ (직선 추가)
- **Stage 5 (TSCOX)**: 25%+ (십자교차로 추가)
- **Stage 6 (TSCOXrR)**: 20%+ (램프 추가)

---

## 🚀 실행 가이드

### 전체 학습 (Stage 1→6)
```bash
python train_curriculum.py
```
- 약 5시간 소요 (1,550k 스텝)
- 모든 Stage를 순차적으로 학습

### 특정 Stage만 학습
```bash
# Stage 1만
python train_curriculum.py --start_stage 1 --end_stage 1

# Stage 2→4
python train_curriculum.py --start_stage 2 --end_stage 4

# 최종 Stage만 (Stage 6)
python train_curriculum.py --start_stage 6 --end_stage 6
```

### 평가
```bash
# 전체 평가 (Stage 1-6)
python evaluate_curriculum.py

# 특정 Stage만
python evaluate_curriculum.py --stages 1 2 3
python evaluate_curriculum.py --stages 6  # 최종 Stage만

# 에피소드 수 변경
python evaluate_curriculum.py --n-episodes 20

# 다른 시드로 일반화 테스트
python evaluate_curriculum.py --seed 2000
```

### 학습 모니터링
```bash
# TensorBoard (별도 터미널)
tensorboard --logdir logs
# http://localhost:6006 접속

# 학습 완료 후 체크
python check_training.py

# 학습 곡선 시각화
python visualize.py
```

---

## 🔍 트러블슈팅

### 문제 1: 로터리에서 멈춤
**원인**: 엔트로피가 너무 낮아서 탐험 부족
**해결**: `ent_coef: "auto"` 설정 (자동 엔트로피 조절)

### 문제 2: Stage 1에서 낮은 성공률
**원인**: 로터리가 가장 어려운 요소
**해결**:
- 학습 시간 증가: 150k → 200k
- 트래픽 감소: 0.05 유지 (더 낮출 수도 있음)
- 추가 학습: `python train_curriculum.py --start_stage 1 --end_stage 1`

### 문제 3: 학습 시작 직후 느림
**원인**: `learning_starts: 5000` (5k 스텝 동안 랜덤 행동으로 Buffer 채움)
**해결**: 정상 동작입니다. 5k 스텝 이후 학습 시작됩니다.

### 문제 4: Replay Buffer 로드 실패
**원인**: 이전 Stage Buffer 파일이 없음
**해결**:
- 이전 Stage부터 순차적으로 학습: `python train_curriculum.py --start_stage 1`
- Buffer는 자동으로 생성/저장됨

### 문제 5: 모델 로드 실패
**원인**: 이전 Stage 모델이 없음
**증상**: `FileNotFoundError: models/sac_stage1_roundabout.zip`
**해결**:
```bash
# Stage 1부터 순차적으로 학습
python train_curriculum.py --start_stage 1 --end_stage 2
```

### 문제 6: 높은 충돌률
**원인**: 트래픽 밀도가 높거나 엔트로피가 너무 높음
**해결**:
- 트래픽 감소: `config.py`에서 `traffic_density` 낮추기
- 추가 학습: 해당 Stage 재학습
- 보상 함수 조정 (선택사항)

---

## 📁 커리큘럼 관련 파일 구조

### **핵심 파일 (3개)**

| 파일명 | 역할 | 주요 기능 |
|--------|------|-----------|
| **config.py** | 설정 | STAGE1~6 환경/학습 설정 정의 |
| **train_curriculum.py** | 학습 | 6단계 순차 학습 + Transfer Learning + Replay Buffer |
| **evaluate_curriculum.py** | 평가 | Stage별 성능 평가 |

### **지원 파일**

| 파일명 | 역할 | 주요 기능 |
|--------|------|-----------|
| **check_training.py** | 모니터링 | 학습 진행 상황 확인 |
| **visualize.py** | 시각화 | 학습 곡선 그래프 생성 |
| **Drive_Record.py** | 녹화 | 주행 GIF 생성 |
| agents/rl_agent.py | 유틸 | SAC 모델 생성 |
| envs/metadrive_env.py | 유틸 | MetaDrive 환경 생성 |
| utils/path_utils.py | 유틸 | 모델/로그 경로 관리 |

### **디렉토리 구조**

```
highway_project/
├── config.py                                    # 📝 전체 설정
│   ├── STAGE1_ENV_CONFIG                       # Stage 1: O (로터리만)
│   ├── STAGE2_ENV_CONFIG                       # Stage 2: OC (로터리+곡선)
│   ├── STAGE3_ENV_CONFIG                       # Stage 3: TOC (T교차로+로터리+곡선)
│   ├── STAGE4_ENV_CONFIG                       # Stage 4: TOCS (직선 추가)
│   ├── STAGE5_ENV_CONFIG                       # Stage 5: TSCOX (십자교차로 추가)
│   ├── STAGE6_ENV_CONFIG                       # Stage 6: TSCOXrR (램프 추가)
│   ├── STAGE1_TRAINING                         # Stage 1 학습 (150k)
│   ├── STAGE2_TRAINING                         # Stage 2 학습 (250k)
│   ├── STAGE3_TRAINING                         # Stage 3 학습 (250k)
│   ├── STAGE4_TRAINING                         # Stage 4 학습 (300k)
│   ├── STAGE5_TRAINING                         # Stage 5 학습 (300k)
│   ├── STAGE6_TRAINING                         # Stage 6 학습 (300k)
│   ├── SAC_CONFIG                              # SAC 알고리즘 설정
│   └── TRAIN_SEEDS                             # 학습용 시드 [1409, 2824, 5506, 6339, 8576, 4806]
│
├── train_curriculum.py                          # 🎓 커리큘럼 학습 스크립트
├── evaluate_curriculum.py                       # 📊 커리큘럼 평가 스크립트
├── check_training.py                            # 🔍 학습 진행 확인
├── visualize.py                                 # 📈 학습 곡선 시각화
├── Drive_Record.py                              # 🎬 주행 GIF 녹화
│
├── agents/
│   └── rl_agent.py                              # 🤖 SAC 모델 생성
├── envs/
│   └── metadrive_env.py                         # 🌍 MetaDrive 환경 생성
├── utils/
│   └── path_utils.py                            # 📂 경로 관리
│
├── models/                                      # 🗂️ 학습된 모델
│   ├── sac_stage1_roundabout.zip               # Stage 1 모델 (O)
│   ├── sac_stage1_roundabout_replay_buffer.pkl # Stage 1 Buffer
│   ├── sac_stage2_roundabout_curve.zip         # Stage 2 모델 (OC)
│   ├── sac_stage2_roundabout_curve_replay_buffer.pkl  # Stage 2 Buffer
│   ├── sac_stage3_toc.zip                      # Stage 3 모델 (TOC)
│   ├── sac_stage3_toc_replay_buffer.pkl        # Stage 3 Buffer
│   ├── sac_stage4_final.zip                    # Stage 4 모델 (TOCS)
│   ├── sac_stage4_final_replay_buffer.pkl      # Stage 4 Buffer
│   ├── sac_stage5_intersection.zip             # Stage 5 모델 (TSCOX)
│   ├── sac_stage5_intersection_replay_buffer.pkl  # Stage 5 Buffer
│   ├── sac_stage6_ramps.zip                    # Stage 6 최종 모델 (TSCOXrR)
│   └── sac_stage6_ramps_replay_buffer.pkl      # Stage 6 Buffer
│
└── logs/                                        # 📊 TensorBoard 로그
    ├── stage1/
    ├── stage2/
    ├── stage3/
    ├── stage4/
    ├── stage5/
    └── stage6/
```

---

## 🔧 파일별 상세 설명

### 1. **config.py** - 설정 파일
**역할**: 모든 Stage의 환경/학습 설정을 정의하는 중앙 설정 파일

**주요 내용**:
```python
# Stage별 환경 설정 (6개)
STAGE1_ENV_CONFIG = {
    "map": "O",                    # 로터리만
    "start_seed": 1000,
    "num_scenarios": 1,
    "traffic_density": 0.05,
    "horizon": 1500,
    # ... 센서 설정 등
}

STAGE2_ENV_CONFIG = {"map": "OC", ...}     # 로터리 + 곡선
STAGE3_ENV_CONFIG = {"map": "TOC", ...}    # T교차로 + 로터리 + 곡선
STAGE4_ENV_CONFIG = {"map": "TOCS", ...}   # 직선 추가
STAGE5_ENV_CONFIG = {"map": "TSCOX", ...}  # 십자교차로 추가
STAGE6_ENV_CONFIG = {"map": "TSCOXrR", ...}  # 램프 추가

# Stage별 학습 설정 (6개)
STAGE1_TRAINING = {
    "total_timesteps": 150000,
    "save_freq": 75000,
    "model_name": "sac_stage1_roundabout",
}
# ... STAGE2~6_TRAINING

# 학습용 시드 (다양한 난이도)
TRAIN_SEEDS = [1409, 2824, 5506, 6339, 8576, 4806]

# SAC 알고리즘 설정
SAC_CONFIG = {
    "learning_rate": 3e-4,
    "buffer_size": 500000,      # Off-policy 핵심 (커리큘럼 대응)
    "learning_starts": 5000,
    "batch_size": 256,
    "gamma": 0.99,              # 장기 보상 중시
    "ent_coef": "auto",         # 자동 엔트로피 조절
    # ...
}
```

**수정 방법**:
- Stage별 학습 시간 조정: `STAGE*_TRAINING["total_timesteps"]`
- 맵 변경: `STAGE*_ENV_CONFIG["map"]`
- 트래픽 밀도: `STAGE*_ENV_CONFIG["traffic_density"]`
- 시나리오 수: `STAGE*_ENV_CONFIG["num_scenarios"]`

---

### 2. **train_curriculum.py** - 학습 스크립트
**역할**: 6단계 커리큘럼 학습을 자동화하는 메인 학습 스크립트

**주요 기능**:
1. **순차 학습**: Stage 1 → 2 → 3 → 4 → 5 → 6
2. **Transfer Learning**: 이전 Stage 모델 로드 후 이어서 학습
3. **Replay Buffer 관리**: Off-policy 핵심 - Buffer 저장/로드/삭제
4. **메모리 효율화**: 이전 Stage Buffer 자동 삭제

**핵심 코드**:
```python
# 이전 모델 + Buffer 로드 (Transfer Learning)
if prev_model_path and os.path.exists(prev_model_path):
    model = SAC.load(prev_model_path, env=env)

    # Replay Buffer 로드 (Off-Policy 핵심!)
    buffer_path = prev_model_path.replace(".zip", "_replay_buffer.pkl")
    if os.path.exists(buffer_path):
        model.load_replay_buffer(buffer_path)
        print(f"Buffer 크기: {model.replay_buffer.size():,} transitions")

        # 디스크 절약: 이전 Buffer 삭제
        os.remove(buffer_path)
else:
    # 새 모델 생성 (Stage 1)
    model = SAC("MlpPolicy", env, **SAC_CONFIG)

# 학습 (Buffer 유지)
model.learn(
    total_timesteps=config["total_timesteps"],
    reset_num_timesteps=False,  # Buffer 초기화 방지!
    progress_bar=True
)

# 모델 + Buffer 저장
model.save(save_path)
model.save_replay_buffer(buffer_path)
```

**사용 예시**:
```bash
# 전체 학습 (Stage 1→6)
python train_curriculum.py

# 특정 Stage만 학습
python train_curriculum.py --start_stage 3 --end_stage 5

# 최종 Stage만 재학습
python train_curriculum.py --start_stage 6 --end_stage 6
```

---

### 3. **evaluate_curriculum.py** - 평가 스크립트
**역할**: 각 Stage 모델을 해당 맵에서 평가하여 성능 측정

**주요 기능**:
1. **Stage별 평가**: 각 모델을 해당 ENV_CONFIG로 평가
2. **통계 생성**: 성공률, 보상, 충돌률, 이탈률 등
3. **결과 저장**: JSON 형식으로 평가 결과 저장

**핵심 코드**:
```python
def evaluate_model(model_path, env_config, n_episodes=10):
    # 환경 생성 (Stage에 맞는 맵)
    env = DummyVecEnv([make_env(config=env_config)])

    # 모델 로드
    model = SAC.load(model_path, env=env)

    # 평가
    for episode in range(n_episodes):
        obs = env.reset()
        done = False
        total_reward = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            total_reward += reward[0]

        # 성공/실패 기록
        if info[0].get("arrive_dest", False):
            success_count += 1

    return {
        "success_rate": success_count / n_episodes,
        "mean_reward": np.mean(episode_rewards),
        # ...
    }
```

**사용 예시**:
```bash
# 전체 Stage 평가
python evaluate_curriculum.py

# 특정 Stage만 평가
python evaluate_curriculum.py --stages 3 4 5

# 에피소드 수 변경
python evaluate_curriculum.py --n-episodes 20

# 특정 시드로 평가
python evaluate_curriculum.py --seed 2000
```

---

### 4. **check_training.py** - 학습 진행 확인
**역할**: TensorBoard 로그를 읽어서 학습 진행 상황을 콘솔에 출력

**출력 예시**:
```
📊 커리큘럼 학습 진행 상황
============================================================
Stage 1 (O):      ✅ 완료 (150,000 / 150,000 스텝)
Stage 2 (OC):     ✅ 완료 (250,000 / 250,000 스텝)
Stage 3 (TOC):    🔄 학습 중 (180,000 / 250,000 스텝) - 72%
Stage 4 (TOCS):   ⏸️  대기 중 (0 / 300,000 스텝)
Stage 5 (TSCOX):  ⏸️  대기 중 (0 / 300,000 스텝)
Stage 6 (TSCOXrR): ⏸️  대기 중 (0 / 300,000 스텝)
============================================================
전체 진행률: 38% (580,000 / 1,550,000 스텝)
```

**사용 예시**:
```bash
python check_training.py
```

---

### 5. **visualize.py** - 학습 곡선 시각화
**역할**: TensorBoard 로그에서 학습 곡선 그래프를 생성

**생성되는 그래프**:
- Stage별 평균 보상 변화
- 성공률 추이
- Actor/Critic Loss
- 엔트로피 계수 변화

**출력 파일**: `results/curriculum_training_curves.png`

**사용 예시**:
```bash
python visualize.py --logdir logs
```

---

### 6. **Drive_Record.py** - GIF 녹화
**역할**: 학습된 모델의 주행 영상을 GIF로 녹화

**주요 기능**:
1. **단일 에피소드 녹화**: 한 시드에서 GIF 생성
2. **여러 에피소드 녹화**: 같은 시드에서 여러 번 실행
3. **시드 비교 녹화**: 여러 시드를 각각 녹화
4. **성공 vs 실패 비교**: 성공 시드와 실패 시드 비교

**사용 예시 (CLI)**:

```bash
# 1. 단일 에피소드 녹화
python Drive_Record.py \
    --model models/sac_stage5_final.zip \
    --seed 1000 \
    --output gifs/stage5.gif \
    --max-steps 2000

# 2. 여러 에피소드 녹화 (같은 시드에서 3번)
python Drive_Record.py \
    --model models/sac_stage5_final.zip \
    --seed 1000 \
    --episodes 3
# 출력: gifs/seed_1000_ep1.gif, ep2.gif, ep3.gif

# 3. 여러 시드 비교 녹화
python Drive_Record.py \
    --model models/sac_stage5_final.zip \
    --seeds 1000 2000 3000 \
    --compare
# 출력: gifs/comparison/seed_1000.gif, seed_2000.gif, seed_3000.gif

# 4. 성공 vs 실패 비교
python Drive_Record.py \
    --model models/sac_stage5_final.zip \
    --success 1000 \
    --failure 2679
# 출력: gifs/comparison/success.gif, failure.gif
```

**Stage별 녹화 예시**:
```bash
# Stage 1 (O 로터리)
python Drive_Record.py --model models/sac_stage1_roundabout.zip --seed 1000

# Stage 3 (TOC 맵)
python Drive_Record.py --model models/sac_stage3_toc.zip --seed 1000

# Stage 6 (TSCOXrR 맵 - 최종)
python Drive_Record.py --model models/sac_stage6_ramps.zip --seed 1000 --max-steps 2000
```

**⚠️ 주의사항**:
- 환경 설정 확인 필요
- Stage별 맵에 맞는 환경 설정 사용
- 필요 시 Drive_Record.py에서 환경 설정 변경:
  ```python
  # 예: Stage 6용 환경 설정
  from config import STAGE6_ENV_CONFIG
  env_config = STAGE6_ENV_CONFIG.copy()
  ```

---

## 🚀 커리큘럼 작업 흐름

### **단계별 실행 순서**

```
1️⃣ 학습 시작
   python train_curriculum.py

   ↓ (학습 중 모니터링)

2️⃣ 진행 확인 (별도 터미널)
   python check_training.py

   또는

   tensorboard --logdir logs
   # http://localhost:6006 접속

   ↓ (학습 완료 후)

3️⃣ 평가
   python evaluate_curriculum.py

   ↓ (평가 완료 후)

4️⃣ 시각화
   python visualize.py

   ↓ (선택 사항)

5️⃣ GIF 녹화
   # Drive_Record.py 수정 후
   python Drive_Record.py
```

---

## 📊 파일 간 데이터 흐름

```
┌─────────────────────────────────────────────────────────┐
│                     config.py                           │
│  (STAGE1~6_ENV_CONFIG, STAGE1~6_TRAINING, SAC_CONFIG)  │
└───────────────┬─────────────────────────────────────────┘
                │
                ↓ (설정 로드)
┌─────────────────────────────────────────────────────────┐
│              train_curriculum.py                        │
│  • Stage 1→6 순차 학습 (O→OC→TOC→TOCS→TSCOX→TSCOXrR)  │
│  • 이전 모델 + Buffer 로드 (Transfer Learning)           │
│  • Buffer 자동 삭제 (메모리 효율화)                      │
│  • agents/rl_agent.py → SAC 모델 생성                   │
│  • envs/metadrive_env.py → 환경 생성                    │
└───────────────┬─────────────────────────────────────────┘
                │
                ↓ (모델 + Buffer 저장)
┌─────────────────────────────────────────────────────────┐
│              models/ & logs/                            │
│  • sac_stage*.zip (모델 파일)                            │
│  • sac_stage*_replay_buffer.pkl (Buffer)                │
│  • logs/stage*/ (TensorBoard 로그)                       │
└───────────────┬─────────────────────────────────────────┘
                │
                ├──→ evaluate_curriculum.py (평가)
                │       ↓
                │    콘솔 출력 (성공률, 보상, 충돌률 등)
                │
                ├──→ check_training.py (진행 확인)
                │       ↓
                │    콘솔 출력 (Stage별 진행률)
                │
                ├──→ visualize.py (시각화)
                │       ↓
                │    학습 곡선 그래프
                │
                └──→ Drive_Record.py (녹화)
                        ↓
                     gifs/*.gif
```

---

## 🎓 핵심 개념

### 1. 커리큘럼 러닝 (Curriculum Learning)
- 어려운 것(로터리)부터 시작해 점진적으로 요소 추가
- 로터리 우선 전략: Bottom-up 접근
- 복잡한 자율주행 환경에 효과적

### 2. 전이 학습 (Transfer Learning)
- 이전 Stage에서 배운 지식을 다음 Stage로 전달
- 모델 가중치 + Replay Buffer 모두 전이
- 학습 시간 단축 + 성능 향상
- `SAC.load(prev_model)` + `load_replay_buffer()` 사용

### 3. Off-Policy Learning (SAC)
- Replay Buffer에 과거 경험 저장
- 이전 Stage 경험을 다음 Stage에서 재사용
- 높은 샘플 효율성
- 메모리 효율: 이전 Buffer 자동 삭제

### 4. 엔트로피 자동 조절
- `ent_coef: "auto"` 사용
- 초기: 높은 엔트로피 (탐험)
- 후기: 낮은 엔트로피 (활용)
- 로터리에서 멈춤 문제 해결

---

## ✅ 체크리스트

### 학습 전
- [ ] `config.py`에서 Stage 1-6 설정 확인
- [ ] SAC_CONFIG 확인 (buffer_size: 500k, ent_coef: "auto")
- [ ] 센서 설정 확인 (lidar: 72, distance: 50)

### 학습 중
- [ ] Progress bar 확인
- [ ] TensorBoard 모니터링 (선택): `tensorboard --logdir logs`
- [ ] 로그 확인 (`logs/stage1-6/` 디렉토리)
- [ ] Buffer 저장 확인 (`*_replay_buffer.pkl`)

### 학습 후
- [ ] `python check_training.py` 실행
- [ ] 각 Stage 모델 저장 확인 (Stage 1-6)
- [ ] `python evaluate_curriculum.py` 실행

### 평가
- [ ] 각 Stage를 해당 맵에서 평가
- [ ] 성공률 목표 달성 확인
- [ ] 최종 Stage 6 성능 확인 (20%+)

---

## 📚 참고 자료

### 성공률 기준 (로터리 우선 전략)
- **Stage 1 (O)**: 70%+ (로터리만)
- **Stage 2 (OC)**: 60%+ (로터리+곡선)
- **Stage 3 (TOC)**: 50%+ (T교차로 추가)
- **Stage 4 (TOCS)**: 30%+ (직선 추가)
- **Stage 5 (TSCOX)**: 25%+ (십자교차로)
- **Stage 6 (TSCOXrR)**: 20%+ (램프 추가)

### 학습 시간 (로터리 우선 전략)
- Stage 1: ~30분 (150k)
- Stage 2: ~50분 (250k)
- Stage 3: ~50분 (250k)
- Stage 4: ~60분 (300k)
- Stage 5: ~60분 (300k)
- Stage 6: ~60분 (300k)
- **총**: ~5시간 (1,550k 스텝)

### 모델 크기
- 각 .zip 파일: ~5-10MB
- 총 6개 모델: ~30-60MB

---

## 🎯 최종 목표

**Stage 6 모델이 TSCOXrR 맵에서 20%+ 성공률 달성!**

로터리 우선 커리큘럼 전략으로 복잡한 자율주행 환경 정복! 🎉

### 일반화 테스트
최종 모델은 학습하지 않은 시드에서도 평가:
```bash
# 학습 시드: 1000 (모든 Stage)
# 테스트 시드: TEST_SEEDS [2679, 3286, 4657, 5012, 9935]

python evaluate_curriculum.py --stages 6 --seed 2679
python evaluate_curriculum.py --stages 6 --seed 4657
```

목표: 다른 시드에서도 15-20% 성공률 유지 (일반화 능력 검증)

---

## 📚 빠른 참조

### **명령어 치트시트**

```bash
# ========== 학습 ==========
# 전체 학습 (Stage 1→6)
python train_curriculum.py

# Stage 3부터 학습
python train_curriculum.py --start_stage 3 --end_stage 6

# Stage 6만 재학습
python train_curriculum.py --start_stage 6 --end_stage 6


# ========== 평가 ==========
# 전체 평가 (Stage 1-6)
python evaluate_curriculum.py

# 특정 Stage만
python evaluate_curriculum.py --stages 1 2 3
python evaluate_curriculum.py --stages 6  # 최종 Stage만

# 에피소드 수 변경
python evaluate_curriculum.py --n-episodes 20

# 다른 시드로 일반화 테스트
python evaluate_curriculum.py --stages 6 --seed 2679


# ========== 모니터링 ==========
# 학습 진행 확인
python check_training.py

# TensorBoard (실시간)
tensorboard --logdir logs
# http://localhost:6006 접속

# 학습 곡선 시각화
python visualize.py


# ========== 녹화 ==========
# GIF 녹화
python Drive_Record.py
```

---

### **파일 빠른 찾기**

| 하고 싶은 일 | 파일 | 명령어 |
|-------------|------|--------|
| 학습 시작 | train_curriculum.py | `python train_curriculum.py` |
| Stage 설정 변경 | config.py | STAGE*_ENV_CONFIG, STAGE*_TRAINING 수정 |
| 성능 평가 | evaluate_curriculum.py | `python evaluate_curriculum.py` |
| 학습 진행 확인 | check_training.py | `python check_training.py` |
| 학습 곡선 보기 | visualize.py | `python visualize.py` |
| 주행 영상 보기 | Drive_Record.py | `python Drive_Record.py` |
| 모델 확인 | models/ | `ls models/sac_stage*.zip` |
| 로그 확인 | logs/ | `tensorboard --logdir logs` |

---

### **Stage별 설정 요약 (로터리 우선 전략)**

| Stage | 맵 | 학습 시간 | 시나리오 | 목표 성공률 | Horizon |
|-------|-----|----------|---------|------------|---------|
| 1 | O | 150k (~30분) | 1 | 70%+ | 1500 |
| 2 | OC | 250k (~50분) | 5 | 60%+ | 1500 |
| 3 | TOC | 250k (~50분) | 10 | 50%+ | 1500 |
| 4 | TOCS | 300k (~60분) | 20 | 30%+ | 1500 |
| 5 | TSCOX | 300k (~60분) | 30 | 25%+ | 1500 |
| 6 | TSCOXrR | 300k (~60분) | 50 | 20%+ | 1500 |

**총 학습 시간**: ~5시간 (1,550k 스텝)

---

### **주요 개념 한눈에 보기**

| 개념 | 설명 | 관련 코드 |
|------|------|----------|
| **Roundabout-First** | 로터리 우선 전략 (어려운 것→쉬운 것 추가) | O→OC→TOC→TOCS |
| **Transfer Learning** | 모델 + Buffer 전이 학습 | `SAC.load()` + `load_replay_buffer()` |
| **Off-Policy** | Replay Buffer로 경험 재사용 | `buffer_size: 500k` |
| **Auto Entropy** | 자동 탐험-활용 균형 | `ent_coef: "auto"` |
| **Memory Efficient** | 이전 Buffer 자동 삭제 | `os.remove(buffer_path)` |

---

### **트러블슈팅 체크리스트**

- [ ] **로터리에서 멈춤** → `ent_coef: "auto"` 확인
- [ ] **Stage 1 낮은 성공률** → 학습 시간 증가 또는 트래픽 감소
- [ ] **모델 로드 실패** → 이전 Stage 모델 존재 여부 확인
- [ ] **Buffer 로드 실패** → `*_replay_buffer.pkl` 파일 확인
- [ ] **높은 충돌률** → 트래픽 밀도 낮추기 또는 추가 학습
- [ ] **학습이 느림** → `learning_starts: 5000` 후 시작 (정상)

---

**작성일**: 2025-01-17
**전략**: 로터리 우선 커리큘럼 (O→OC→TOC→TOCS→TSCOX→TSCOXrR)
**최종 수정**: 2025-12-03
**버전**: 3.0 (로터리 우선 전략으로 전면 개편)
