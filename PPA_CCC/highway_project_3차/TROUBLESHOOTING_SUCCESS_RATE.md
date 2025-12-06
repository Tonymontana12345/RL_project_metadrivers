# Success Rate 그래프 문제 해결 가이드

## 🔍 문제 증상

`visualize.py` 실행 시 "Success Rate Across Seeds" 그래프가 비어있거나 막대가 보이지 않습니다.

![문제 예시](https://i.imgur.com/example.png)

---

## 💡 원인 분석

### 주요 원인

1. **성공률이 0%인 경우**
   - 모든 시드에서 에이전트가 목적지에 도달하지 못함
   - 막대 높이가 0이라서 시각적으로 보이지 않음

2. **학습이 불충분한 경우**
   - 학습 스텝이 너무 적음 (예: < 50K 스텝)
   - 에이전트가 아직 목적지 도달 방법을 학습하지 못함

3. **평가 데이터가 없는 경우**
   - `results/evaluation_results.json` 파일이 없거나 손상됨
   - 평가를 실행하지 않았거나 실패함

---

## 🛠️ 해결 방법

### 1단계: 평가 결과 확인

```bash
# 디버깅 스크립트 실행
python debug_results.py --results results/evaluation_results.json
```

**출력 예시**:
```
🔍 평가 결과 디버깅
============================================================
✅ 파일 로드 성공: results/evaluation_results.json

📋 기본 정보:
  모델: models/ppo_fixed_seed_1000.zip
  테스트 시드: [1000, 2000, 2001, 2002, 2003, 2004]
  에피소드/시드: 20

📊 시드별 결과 (6개):

  시드 1000:
    평균 보상: 15.23
    성공률: 0.0%      ← 문제!
    충돌률: 45.0%
    도로 이탈률: 55.0%
```

### 2단계: 학습 상태 확인

```bash
# TensorBoard로 학습 곡선 확인
tensorboard --logdir logs/
# 브라우저에서 http://localhost:6006 접속
```

**확인 사항**:
- `rollout/ep_rew_mean`이 증가하는지 확인
- 보상이 15 이상으로 올라가는지 확인
- 학습이 수렴했는지 확인

### 3단계: 충분한 학습 실행

```bash
# 고정 시드로 100K 스텝 학습
python train.py --mode fixed --algorithm ppo

# 또는 실험 프로토콜 사용
python train.py --experiment exp1_fixed_seed --algorithm ppo
```

**학습 시간**:
- PPO: 약 1-2시간 (100K 스텝)
- SAC: 약 30-60분 (더 빠름)
- TD3: 약 40-80분

### 4단계: 재평가 및 시각화

```bash
# 학습된 모델 평가
python evaluate.py --model models/ppo_fixed_seed_1000.zip --episodes 20

# 결과 시각화
python visualize.py --results results/evaluation_results.json
```

---

## ✅ 수정 사항

### 코드 개선

`visualize.py`를 수정하여 **성공률이 0%일 때도 값이 표시**되도록 개선했습니다:

```python
# 막대 위에 정확한 값 표시 (0%일 때도 보이도록)
for i, (bar, rate) in enumerate(zip(bars, success_rates)):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
            f'{rate:.1f}%',
            ha='center', va='bottom', fontsize=10)
```

**효과**:
- 성공률이 0%여도 "0.0%" 텍스트가 표시됨
- 어떤 시드에서 문제가 있는지 명확히 파악 가능

---

## 📊 예상 성공률

### 학습 단계별 성공률

| 학습 스텝 | 학습 시드 (1000) | 테스트 시드 (2000-2004) |
|----------|-----------------|----------------------|
| 10K      | 0-10%           | 0-5%                 |
| 50K      | 30-50%          | 20-40%               |
| 100K     | 70-85%          | 50-70%               |
| 200K+    | 85-95%          | 60-80%               |

### 알고리즘별 성공률 (100K 스텝 기준)

| 알고리즘 | 학습 시드 | 테스트 시드 평균 |
|---------|----------|---------------|
| PPO     | 80-85%   | 50-60%        |
| SAC     | 85-90%   | 55-65%        |
| TD3     | 82-88%   | 52-62%        |

---

## 🎯 체크리스트

학습 및 평가가 제대로 되었는지 확인하세요:

- [ ] 모델 파일이 `models/` 디렉토리에 있음
- [ ] 학습 로그가 `logs/` 디렉토리에 있음
- [ ] TensorBoard에서 보상이 증가하는 추세를 확인
- [ ] 평가 결과 JSON 파일이 `results/`에 생성됨
- [ ] `debug_results.py`로 성공률 확인
- [ ] 성공률이 0%보다 큼 (최소 10% 이상 권장)

---

## 💬 추가 도움

### 학습이 너무 느린 경우

```python
# config.py 수정
FIXED_SEED_ENV_CONFIG = {
    "decision_repeat": 10,  # 기본 5에서 증가 (속도 2배)
    "use_render": False,    # 렌더링 끄기
}
```

### 빠른 테스트

```bash
# 10K 스텝으로 빠른 테스트
python train.py --mode quick

# 평가
python evaluate.py --model models/ppo_quick_test.zip --episodes 5
```

### 환경 확인

```bash
# 환경이 제대로 작동하는지 확인
python quick_start.py --demo

# 수동으로 운전해보기
python quick_start.py --manual
```

---

## 📞 문의

문제가 계속되면:

1. `debug_results.py` 출력 확인
2. TensorBoard 학습 곡선 스크린샷 확인
3. 로그 파일 확인 (`logs/`)

---

**Happy Debugging! 🐛🔧**
