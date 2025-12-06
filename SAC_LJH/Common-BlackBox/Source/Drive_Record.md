# 주행 영상 GIF 녹화 가이드 🎬

## 📚 기능 소개

`screen_record=True`와 `env.top_down_renderer.generate_gif()`를 사용하여 주행 과정을 GIF로 녹화합니다.

**장점:**
- ✅ 시각적으로 주행 확인
- ✅ 실패 원인 파악 용이
- ✅ 프레젠테이션/리포트용
- ✅ 성공 vs 실패 비교

---

## 🚀 빠른 시작


### 1. 단일 에피소드 녹화
```bash
python evalute.py --model models/sac_fixed_seed_1000.zip --seed 1000 --record-gif
```


### 1. 단일 에피소드 녹화
```bash
python evalute.py --model models/sac_fixed_seed_1000.zip --seed 1000 --record-success
```


-------
## 🚀 빠른 시작


### 1. 단일 에피소드 녹화
```bash
python Drive_Record.py --model models/sac_fixed_seed_1000.zip --seed 1000
```
**결과:** `driving.gif`

### 2. 커스텀 파일명
```bash
python Drive_Record.py --model models/sac_fixed_seed_1000.zip --seed 1000 --output my_driving.gif
```
**결과:** `my_driving.gif`

### 3. 실패하는 시드 녹화 (중요!)
```bash
python Drive_Record.py --model models/sac_fixed_seed_1000.zip --seed 2679 --output failure_seed_2679.gif
```
**결과:** 실패 지점을 GIF로 확인!

---

## 📊 다양한 사용법

### A. 여러 에피소드 녹화
```bash
python Drive_Record.py --model models/sac_fixed_seed_1000.zip --seed 2679 --episodes 3
```
**결과:**
- `gifs/seed_2679_ep1.gif`
- `gifs/seed_2679_ep2.gif`
- `gifs/seed_2679_ep3.gif`

### B. 여러 시드 비교
```bash
python Drive_Record.py --model models/sac_fixed_seed_1000.zip --seeds 1000 2679 4657 --compare
```
**결과:**
- `gifs/comparison/seed_1000.gif`
- `gifs/comparison/seed_2679.gif`
- `gifs/comparison/seed_4657.gif`

### C. 성공 vs 실패 비교 (추천! ⭐)
```bash
python Drive_Record.py --model models/sac_fixed_seed_1000.zip --success 1000 --failure 2679
```
**결과:**
- `gifs/comparison/success.gif` - 성공 사례
- `gifs/comparison/failure.gif` - 실패 사례

### D. 긴 에피소드 녹화
```bash
python Drive_Record.py --model models/sac_fixed_seed_1000.zip --seed 1000 --max-steps 2000
```
**결과:** 최대 2000 스텝까지 녹화

---

## 🎯 실전 활용 예시

### 사례 1: 실패 원인 분석

**목표:** 시드 2679에서 왜 실패하는지 확인

```bash
# 1. 실패 시드 녹화
python record_driving_gif.py --model models/sac_fixed_seed_1000.zip --seed 2679 --output analysis_seed_2679.gif

# 2. GIF 확인
# → 커브에서 도로 이탈하는 장면 발견!
```

### 사례 2: 성공 vs 실패 비교

**목표:** 성공하는 시드와 실패하는 시드 비교

```bash
python record_driving_gif.py --model models/sac_fixed_seed_1000.zip --success 1000 --failure 2679

# 결과:
# - success.gif: 안정적인 주행
# - failure.gif: 커브에서 미끄러짐
```

### 사례 3: 모든 테스트 시드 녹화

**목표:** 평가 결과가 안 좋은 모든 시드 확인

```bash
# evaluation_results.json에서 실패 시드 확인:
# 2679 (성공률 0%), 4657 (성공률 0%), 9935 (성공률 0%)

python record_driving_gif.py --model models/sac_fixed_seed_1000.zip --seeds 2679 4657 9935 --compare

# 결과: gifs/comparison/ 폴더에 3개 GIF 생성
```

### 사례 4: 알고리즘 비교

```bash
# SAC 모델
python record_driving_gif.py --model models/sac_fixed_seed_1000.zip --seed 2679 --output sac_seed_2679.gif

# PPO 모델
python record_driving_gif.py --model models/ppo_fixed_seed_1000.zip --seed 2679 --output ppo_seed_2679.gif

# 결과:
# - sac_seed_2679.gif
# - ppo_seed_2679.gif
# → 두 GIF를 나란히 놓고 비교!
```

---

## 📋 명령어 옵션

| 옵션 | 설명 | 예시 |
|------|------|------|
| `--model` | 모델 경로 (필수) | `--model models/sac.zip` |
| `--seed` | 단일 시드 | `--seed 1000` |
| `--seeds` | 여러 시드 | `--seeds 1000 2679 4657` |
| `--episodes` | 에피소드 수 | `--episodes 3` |
| `--output` | 출력 파일명 | `--output my.gif` |
| `--max-steps` | 최대 스텝 수 | `--max-steps 2000` |
| `--compare` | 비교 모드 | `--compare` |
| `--success` | 성공 시드 | `--success 1000` |
| `--failure` | 실패 시드 | `--failure 2679` |

---

## 🎨 시각화 옵션

스크립트 내부에서 다음 옵션 수정 가능:

```python
env.render(
    mode="topdown",              # 탑다운 뷰
    window=False,                # 창 표시 안 함
    screen_record=True,          # 녹화 활성화
    screen_size=(800, 800),      # 화면 크기 (변경 가능)
    camera_position=(50, 50),    # 카메라 위치 (변경 가능)
    scaling=2,                   # 확대/축소 (변경 가능)
)
```

### 다양한 뷰 설정

#### 1. 넓은 뷰 (전체 맵 보기)
```python
screen_size=(1200, 1200)
camera_position=(80, 80)
scaling=1.5
```

#### 2. 줌인 뷰 (차량 중심)
```python
screen_size=(600, 600)
camera_position=(30, 30)
scaling=3
```

#### 3. 고해상도
```python
screen_size=(1920, 1920)
scaling=2
```

---

## 💡 MetaDrive 문서 예제

문서에서 제공하는 기본 예제:

```python
from metadrive import MetaDriveEnv

env = MetaDriveEnv({"map": "C", "traffic_density": 0.2})
env.reset()

try:
    for step in range(300):
        obs, reward, terminated, truncated, info = env.step([0, 0.5])
        
        env.render(
            mode="topdown",
            window=False,
            screen_record=True,
            screen_size=(700, 870),
            camera_position=(60, -63)
        )
        
        if terminated or truncated:
            break
    
    # GIF 생성
    env.top_down_renderer.generate_gif()
finally:
    env.close()
```

---

## 📁 파일 구조

```
프로젝트/
├── record_driving_gif.py      # GIF 녹화 스크립트
├── driving.gif                # 기본 출력
├── gifs/                      # 여러 에피소드 출력
│   ├── seed_2679_ep1.gif
│   ├── seed_2679_ep2.gif
│   └── seed_2679_ep3.gif
└── gifs/comparison/           # 비교용 출력
    ├── seed_1000.gif
    ├── seed_2679.gif
    ├── success.gif
    └── failure.gif
```

---

## ⚠️ 주의사항

### 1. GIF 파일 크기
- 긴 에피소드는 GIF 파일이 매우 클 수 있음
- 권장: `--max-steps 1000` 이하

### 2. 녹화 속도
- GIF 생성에 시간이 걸릴 수 있음
- 에피소드당 약 10-30초 소요

### 3. 화면 크기
- 너무 크면 파일 크기 증가
- 권장: `(800, 800)` 또는 `(600, 600)`

### 4. 3D 렌더링
- `use_render=False`로 설정 필요
- 3D와 GIF 동시 사용 불가능

---

## 🎓 실전 워크플로우

### Step 1: 평가 결과 확인
```bash
python evaluate.py --model models/sac_fixed_seed_1000.zip
# 결과: seed 2679, 4657, 9935 실패
```

### Step 2: 실패 시드 GIF 녹화
```bash
python record_driving_gif.py --model models/sac_fixed_seed_1000.zip --seeds 2679 4657 9935 --compare
```

### Step 3: GIF 확인 및 분석
```bash
# gifs/comparison/ 폴더 확인
# - seed_2679.gif: 초반 커브에서 도로 이탈
# - seed_4657.gif: 교차로 진입 실패
# - seed_9935.gif: 급커브에서 충돌
```

### Step 4: 개선 및 재평가
```bash
# 학습 개선 후
python train.py --mode fixed

# 다시 녹화
python record_driving_gif.py --model models/sac_improved.zip --seeds 2679 4657 9935 --compare
```

---

## 🎬 프레젠테이션용 활용

### 1. 성공 사례 강조
```bash
python record_driving_gif.py --model models/sac_final.zip --seed 1000 --output presentation_success.gif
```

### 2. 개선 전후 비교
```bash
# Before
python record_driving_gif.py --model models/sac_v1.zip --seed 2679 --output before.gif

# After
python record_driving_gif.py --model models/sac_v2.zip --seed 2679 --output after.gif
```

### 3. 다양한 시나리오 시연
```bash
python record_driving_gif.py --model models/sac_final.zip --seeds 1000 2679 3286 4657 5012 9935 --compare
# → 6개 GIF로 다양한 환경 시연
```

---

## 🔗 관련 도구

이미 만들어진 다른 시각화 도구와 함께 사용:

1. **visualize_trajectory.py** - 궤적 분석
2. **record_driving_gif.py** - 주행 영상 (현재)
3. **visualize_maps_topdown.py** - 맵 시각화

**통합 워크플로우:**
```bash
# 1. 맵 확인
python visualize_maps_topdown.py --seed 2679 --save --output seed_2679_map.png

# 2. 주행 영상
python record_driving_gif.py --model models/sac.zip --seed 2679 --output seed_2679_driving.gif

# 3. 궤적 분석
python visualize_trajectory.py --model models/sac.zip --seed 2679 --episodes 5

# 결과:
# - 맵 이미지
# - 주행 GIF
# - 궤적 분석 이미지
# → 완벽한 분석 패키지!
```

---

## ✅ 체크리스트

프로젝트 완료 전 확인:

- [ ] 학습 시드 주행 GIF 확인 (성공 사례)
- [ ] 실패 시드 주행 GIF 확인 (문제 파악)
- [ ] 성공 vs 실패 비교 GIF 생성
- [ ] 리포트/프레젠테이션용 GIF 준비
- [ ] 개선 전후 비교 GIF 준비

---

## 🚀 지금 바로 시작!

```bash
# 가장 간단한 사용법
python record_driving_gif.py --model models/sac_fixed_seed_1000.zip --seed 2679

# 실패 분석
python record_driving_gif.py --model models/sac_fixed_seed_1000.zip --success 1000 --failure 2679
```

주행 과정을 눈으로 확인하면 실패 원인이 명확해집니다! 🎯