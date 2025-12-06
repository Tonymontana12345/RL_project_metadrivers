# 실제 도로 스타일 맵 시각화 가이드

## 🎨 개요

기존의 블록 다이어그램 대신 **실제 도로처럼 보이는 맵**을 생성합니다.

### 차이점 비교

| 특징 | 블록 다이어그램 | 실제 도로 스타일 |
|------|---------------|----------------|
| **스타일** | 간단한 색상 블록 | 실제 도로 렌더링 |
| **색상** | 블록 타입별 색상 | 회색 도로 + 흰색/노란색 차선 |
| **배경** | 흰색 | 잔디 색 (#3a5f0b) |
| **세부사항** | 블록 이름만 | 차선, 도로 가장자리, 시작/목적지 |
| **용도** | 빠른 구조 파악 | 시각적 이해, 프레젠테이션 |

---

## 🚀 사용법

### 기본 실행

```bash
# 모든 시드 (1000, 2000-2004) 실제 도로 스타일로 시각화
./venv_pgdrive/bin/python visualize_maps_3d.py
```

**생성되는 파일**:
- `results/seed_1000_realistic.png`
- `results/seed_2000_realistic.png`
- `results/seed_2001_realistic.png`
- `results/seed_2002_realistic.png`
- `results/seed_2003_realistic.png`
- `results/seed_2004_realistic.png`

### 특정 시드만 생성

```bash
# 학습 시드와 첫 번째 테스트 시드만
./venv_pgdrive/bin/python visualize_maps_3d.py --seeds 1000 2000

# 여러 시드 선택
./venv_pgdrive/bin/python visualize_maps_3d.py --seeds 1000 2001 2002
```

---

## 🎨 시각화 스타일 설명

### 실제 도로 스타일 (Realistic)

**특징**:
- 🟢 **배경**: 잔디 색 (#3a5f0b)
- ⬛ **도로 표면**: 어두운 회색 (#3d3d3d)
- ⬜ **도로 가장자리**: 흰색 실선
- 🟡 **차선 중앙선**: 노란색 점선
- 🟢 **시작점**: 초록색 원 (Start)
- 🔴 **목적지**: 빨간색 별 (Goal)

**예시**:
```
┌─────────────────────────────────────┐
│  🌿 잔디 배경                        │
│                                     │
│  🟢 Start ━━━━━━━━━ 🔴 Goal        │
│         ╱╲  도로  ╱╲                │
│        ╱  ╲      ╱  ╲               │
│  ━━━━━    ━━━━━    ━━━━━          │
│  흰색      노란색    흰색             │
│  가장자리  중앙선    가장자리          │
│                                     │
└─────────────────────────────────────┘
```

### 조감도 스타일 (Bird's Eye View)

```bash
# 조감도 스타일로 생성
./venv_pgdrive/bin/python visualize_maps_3d.py --style birdseye
```

**특징**:
- 더 단순화된 스타일
- 차선 경계만 표시
- 빠른 생성 속도

---

## 📊 생성된 맵 예시

### 시드 1000: 1SXXCT
- **구성**: 시작 → 직선 → 교차로 → 교차로 → 커브 → T자 교차로
- **난이도**: 🟢 쉬움 (점수 2)
- **특징**: 교차로 2개 포함

### 시드 2000: 1STCTC
- **구성**: 시작 → 직선 → T자 교차로 → 커브 → T자 교차로 → 커브
- **난이도**: 🟢 쉬움 (점수 4)
- **특징**: 직선과 커브 균형

### 시드 2001: 1CCCOO
- **구성**: 시작 → 커브 → 커브 → 커브 → 로터리 → 로터리
- **난이도**: 🔴 어려움 (점수 14)
- **특징**: 로터리 2개 포함, 가장 어려움

### 시드 2002: 1CTSCS
- **구성**: 시작 → 커브 → T자 교차로 → 직선 → 커브 → 직선
- **난이도**: 🟢 쉬움 (점수 4)
- **특징**: 균형잡힌 구성

### 시드 2003: 1OOSSC
- **구성**: 시작 → 로터리 → 로터리 → 직선 → 직선 → 커브
- **난이도**: 🟠 보통 (점수 10)
- **특징**: 로터리 2개 포함

### 시드 2004: 1OOCCS
- **구성**: 시작 → 로터리 → 로터리 → 커브 → 커브 → 직선
- **난이도**: 🔴 어려움 (점수 12)
- **특징**: 로터리와 커브 많음

---

## 🎯 활용 방법

### 1. 프레젠테이션 자료

```bash
# 모든 시드 생성
./venv_pgdrive/bin/python visualize_maps_3d.py

# PowerPoint/Keynote에 삽입
# results/seed_*_realistic.png 파일 사용
```

**장점**:
- 시각적으로 이해하기 쉬움
- 전문적인 외관
- 맵 구조 명확히 전달

### 2. 논문/리포트 작성

```bash
# 고해상도 이미지 생성 (코드 수정)
# visualize_maps_3d.py에서 dpi=200 → dpi=300
```

**사용 예시**:
```latex
\begin{figure}[h]
    \centering
    \includegraphics[width=0.8\textwidth]{seed_1000_realistic.png}
    \caption{Training Seed 1000 Map Structure}
\end{figure}
```

### 3. 맵 구조 분석

```bash
# 1. 실제 도로 스타일로 맵 생성
./venv_pgdrive/bin/python visualize_maps_3d.py

# 2. 블록 다이어그램과 비교
./venv_pgdrive/bin/python visualize_maps.py --save

# 3. 난이도 분석
./venv_pgdrive/bin/python visualize_maps.py --difficulty --save
```

**분석 포인트**:
- 도로 길이와 복잡도
- 커브의 급격함
- 교차로/로터리 위치
- 시작점에서 목적지까지 경로

### 4. 학습 결과와 비교

```bash
# 1. 맵 시각화
./venv_pgdrive/bin/python visualize_maps_3d.py

# 2. 평가 실행
./venv_pgdrive/bin/python evaluate.py --model models/ppo_fixed_seed_1000.zip

# 3. 결과 비교
# - 시드 1000 (학습): 성공률 vs 맵 구조
# - 시드 2001 (어려움): 성공률 vs 로터리 2개
```

---

## 🔧 커스터마이징

### 색상 변경

`visualize_maps_3d.py` 파일 수정:

```python
# 배경색 변경
ax.set_facecolor('#3a5f0b')  # 잔디 색
# → ax.set_facecolor('#87ceeb')  # 하늘색

# 도로 색상 변경
facecolor='#3d3d3d'  # 어두운 회색
# → facecolor='#2c2c2c'  # 더 어두운 회색

# 차선 색상 변경
color='#ffd700'  # 노란색
# → color='white'  # 흰색
```

### 해상도 변경

```python
# 기본 해상도
fig, ax = plt.subplots(figsize=(16, 16))

# 더 큰 해상도
fig, ax = plt.subplots(figsize=(20, 20))

# 저장 시 DPI 변경
plt.savefig(save_path, dpi=200, ...)
# → plt.savefig(save_path, dpi=300, ...)  # 더 선명함
```

### 추가 요소 표시

```python
# 차량 위치 표시
ax.plot(vehicle_x, vehicle_y, 'bo', markersize=10, label='Vehicle')

# 장애물 표시
ax.plot(obstacle_x, obstacle_y, 'rx', markersize=15, label='Obstacle')

# 경로 표시
ax.plot(path_x, path_y, 'g--', linewidth=2, label='Planned Path')
```

---

## 📁 파일 구조

```
results/
├── seed_1000_realistic.png    # 학습 시드 (실제 도로 스타일)
├── seed_2000_realistic.png    # 테스트 시드 1
├── seed_2001_realistic.png    # 테스트 시드 2 (가장 어려움)
├── seed_2002_realistic.png    # 테스트 시드 3
├── seed_2003_realistic.png    # 테스트 시드 4
├── seed_2004_realistic.png    # 테스트 시드 5
├── seed_maps_comparison.png   # 블록 다이어그램 비교
└── seed_difficulty_comparison.png  # 난이도 비교 그래프
```

---

## 💡 팁

### 1. 두 스타일 모두 생성

```bash
# 블록 다이어그램 (빠른 구조 파악)
./venv_pgdrive/bin/python visualize_maps.py --save

# 실제 도로 스타일 (시각적 이해)
./venv_pgdrive/bin/python visualize_maps_3d.py
```

### 2. 이미지 확인

```bash
# macOS
open results/seed_1000_realistic.png

# 모든 이미지 한번에 보기
open results/seed_*_realistic.png
```

### 3. 프레젠테이션 준비

```bash
# 1. 모든 맵 생성
./venv_pgdrive/bin/python visualize_maps_3d.py

# 2. 난이도 분석
./venv_pgdrive/bin/python visualize_maps.py --difficulty --save

# 3. 평가 결과
./venv_pgdrive/bin/python visualize.py --results results/evaluation_results.json
```

**슬라이드 구성**:
1. 맵 구조 소개 (블록 다이어그램)
2. 실제 도로 모습 (실제 도로 스타일)
3. 난이도 분석 (난이도 비교 그래프)
4. 평가 결과 (성공률 그래프)

---

## 🎓 실험 아이디어

### 실험 1: 맵 복잡도와 성능

```bash
# 1. 모든 맵 시각화
./venv_pgdrive/bin/python visualize_maps_3d.py

# 2. 평가
./venv_pgdrive/bin/python evaluate.py --model models/ppo_fixed_seed_1000.zip

# 3. 분석
# - 단순한 맵 (2000, 2002): 성공률 높을 것으로 예상
# - 복잡한 맵 (2001, 2004): 성공률 낮을 것으로 예상
```

### 실험 2: 시각적 경로 계획

```python
# visualize_maps_3d.py 수정하여 에이전트 경로 표시
# 학습된 에이전트가 실제로 어떤 경로를 선택하는지 시각화
```

---

## 🔍 문제 해결

### 문제 1: 이미지가 너무 작음

**해결**:
```python
# visualize_maps_3d.py 수정
fig, ax = plt.subplots(figsize=(16, 16))
# → fig, ax = plt.subplots(figsize=(24, 24))
```

### 문제 2: 도로가 잘 안보임

**해결**:
```python
# 도로 색상 대비 증가
facecolor='#3d3d3d'  # 어두운 회색
# → facecolor='#2c2c2c'  # 더 어두운 회색

# 차선 두께 증가
linewidth=2
# → linewidth=3
```

### 문제 3: 생성 속도가 느림

**해결**:
```bash
# 필요한 시드만 생성
./venv_pgdrive/bin/python visualize_maps_3d.py --seeds 1000

# 또는 블록 다이어그램 사용 (더 빠름)
./venv_pgdrive/bin/python visualize_maps.py
```

---

## 📞 추가 정보

### 관련 파일
- `visualize_maps_3d.py` - 실제 도로 스타일 시각화
- `visualize_maps.py` - 블록 다이어그램 시각화
- `MAP_VISUALIZATION_GUIDE.md` - 맵 시각화 전체 가이드

### 명령어 요약

```bash
# 실제 도로 스타일
./venv_pgdrive/bin/python visualize_maps_3d.py

# 블록 다이어그램
./venv_pgdrive/bin/python visualize_maps.py

# 난이도 분석
./venv_pgdrive/bin/python visualize_maps.py --difficulty --save
```

---

**Happy Visualizing! 🗺️🎨**
