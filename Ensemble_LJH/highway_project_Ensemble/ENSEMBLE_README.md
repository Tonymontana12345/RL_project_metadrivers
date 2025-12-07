# 🤝 앙상블 모델 사용 가이드

TD3와 SAC 모델을 앙상블로 결합하여 더 나은 성능을 얻을 수 있습니다!

## 🚀 빠른 시작

**전체 워크플로우:**
1. 평가 + GIF 녹화 → 2. 결과 시각화 → 3. 분석 완료! 🎉

### 1. 평가 + GIF 녹화 (한번에!)

```bash
python evaluate_ensemble.py \
    --models models/td3_tsco_map_500k.zip models/sac_stage4_final.zip \
    --strategy scenario_based \
    --maps CSTO OSCT \
    --episodes 5 \
    --record-gif \
    --save ensemble_results.json
```

**결과물:**
- `results/ensemble_results.json` - 평가 결과 데이터
- `results/ensemble_results_report.txt` - 텍스트 리포트
- `gifs/ensemble/ensemble_scenario_based_CSTO_seed1000.gif` - CSTO 주행 영상
- `gifs/ensemble/ensemble_scenario_based_OSCT_seed1000.gif` - OSCT 주행 영상

---

### 1-1. 평가만 하기 (GIF 없이)

성능 평가만 빠르게 하고 싶을 때:

```bash
python evaluate_ensemble.py \
    --models models/td3_tsco_map_500k.zip models/sac_stage4_final.zip \
    --strategy scenario_based \
    --maps CSTO OSCT TSCO SCTO \
    --episodes 10 \
    --save ensemble_results.json
```

**결과물:**
- `results/ensemble_results.json` - 평가 결과 데이터
- `results/ensemble_results_report.txt` - 텍스트 리포트 (GIF 없음)

---

### 2. GIF만 빠르게 녹화

특정 맵에서 앙상블 주행을 빠르게 확인:

```bash
python record_ensemble_gif.py \
    --models models/td3_tsco_map_500k.zip models/sac_stage4_final.zip \
    --strategy scenario_based \
    --map CSTO \
    --seed 1000
```

**결과물:**
- `ensemble_driving.gif` - 주행 영상

---

### 3. 여러 전략 비교

같은 맵에서 다양한 앙상블 전략을 비교:

```bash
python record_ensemble_gif.py \
    --models models/td3_tsco_map_500k.zip models/sac_stage4_final.zip \
    --compare-strategies \
    --map CSTO \
    --seed 1000
```

**결과물:**
- `gifs/ensemble/comparison/averaging_CSTO_seed1000.gif`
- `gifs/ensemble/comparison/q_value_weighted_CSTO_seed1000.gif`
- `gifs/ensemble/comparison/scenario_based_CSTO_seed1000.gif`

---

### 4. Q-value 기반 동적 앙상블

Q-value를 기반으로 동적으로 가중치를 조정하는 전략:

```bash
python evaluate_ensemble.py \
    --models models/td3_tsco_map_500k.zip models/sac_stage4_final.zip \
    --strategy q_value_weighted \
    --maps CSTO OSCT TSCO SCTO \
    --episodes 10 \
    --save q_value_results.json
```

**특징:**
- 각 모델의 Q-value를 실시간으로 계산
- Q-value가 높은 모델에 더 큰 가중치 부여
- 상황에 따라 동적으로 모델 선택
- 고정된 가중치 없이 자동 조정

**언제 사용?**
- 시나리오별 가중치를 모를 때
- 동적인 환경에서 적응이 필요할 때
- 모델의 신뢰도를 자동으로 판단하고 싶을 때

---

### 5. 결과 시각화

평가 후 결과를 차트로 시각화:

```bash
# 모든 차트 생성
python visualize_ensemble.py --results results/ensemble_results.json

# 특정 차트만 생성
python visualize_ensemble.py --results results/ensemble_results.json --chart map
python visualize_ensemble.py --results results/ensemble_results.json --chart seed
python visualize_ensemble.py --results results/ensemble_results.json --chart heatmap
python visualize_ensemble.py --results results/ensemble_results.json --chart dashboard
```

**생성되는 차트:**
- `map`: 맵별 성능 비교 (성공률, 평균 보상)
- `seed`: 시드별 성능 분석
- `failure`: 실패 원인 분석
- `heatmap`: 맵-시드 히트맵
- `dashboard`: 종합 대시보드
- `all`: 모든 차트 한번에 생성

**결과물:**
- `results/ensemble_results_map_comparison.png`
- `results/ensemble_results_seed_comparison.png`
- `results/ensemble_results_failure_analysis.png`
- `results/ensemble_results_heatmap.png`
- `results/ensemble_results_dashboard.png`

---

## 📊 앙상블 전략

| 전략 | 설명 | 추천 | 사용법 |
|------|------|------|--------|
| **scenario_based** ⭐ | 맵별로 각 모델에 다른 가중치 부여 | ✅ 권장! | `--strategy scenario_based` |
| **averaging** | 모든 모델의 액션을 평균 | ✅ 안정적 | `--strategy averaging` |
| **q_value_weighted** 🔥 | Q-value에 비례하여 가중치 부여 | 🎯 동적 조정 | `--strategy q_value_weighted` |
| **confidence_weighted** | 불확실성 기반 가중치 | 고급 | `--strategy confidence_weighted` |
| **voting** | 투표 방식 (연속 액션에서는 평균과 유사) | 기본 | `--strategy voting` |

### 전략 선택 가이드

**시나리오별 가중치를 알고 있을 때:**
```bash
--strategy scenario_based
```
- 각 맵에 대한 최적 가중치를 미리 설정
- 가장 높은 성능 보장

**가중치를 모르지만 동적으로 조정하고 싶을 때:**
```bash
--strategy q_value_weighted
```
- Q-value를 기반으로 실시간 가중치 계산
- 자동으로 더 신뢰도 높은 모델 선택
- scenario_based보다 유연하지만 약간 느림

**빠르고 안정적인 결과를 원할 때:**
```bash
--strategy averaging
```
- 단순 평균으로 빠른 계산
- 예측 가능한 안정적인 결과

---

## 💡 기대 효과

scenario_based 전략 사용 시:

| 시나리오 | TD3 단독 | SAC 단독 | 앙상블 |
|---------|----------|----------|--------|
| CSTO | ~80%+ | ~0% | **~64%+** ⬆️ |
| OSCT | ~0% | ~33% | **~23%+** ⬆️ |
| **전체 평균** | ~40% | ~16% | **~43%+** 🎯 |

---

## 📁 파일 구조

```
highway_project/
├── ensemble_agent.py          # 앙상블 에이전트 핵심 구현
├── evaluate_ensemble.py       # 평가 + GIF 녹화 스크립트
├── record_ensemble_gif.py     # GIF 녹화 전용 스크립트
├── ENSEMBLE_GUIDE.md          # 상세 가이드 (이 문서)
└── ENSEMBLE_README.md         # 빠른 시작 가이드
```

---

## 🎬 GIF 녹화 옵션

### 평가 중 자동 녹화

`--record-gif` 플래그 추가:

```bash
# Scenario-based 전략으로 평가 + GIF
python evaluate_ensemble.py \
    --models models/td3_tsco_map_500k.zip models/sac_stage4_final.zip \
    --strategy scenario_based \
    --record-gif                    # 🎬 GIF 녹화 활성화
    --gif-dir gifs/my_ensemble      # 저장 위치 (선택)

# Q-value 전략으로 평가 + GIF
python evaluate_ensemble.py \
    --models models/td3_tsco_map_500k.zip models/sac_stage4_final.zip \
    --strategy q_value_weighted \
    --maps CSTO OSCT \
    --episodes 5 \
    --record-gif
```



### 수동 녹화

더 많은 제어가 필요하면 `record_ensemble_gif.py` 사용:

```bash
# 여러 시나리오 한번에
python record_ensemble_gif.py \
    --models models/td3_tsco_map_500k.zip models/sac_stage4_final.zip \
    --maps CSTO OSCT TSCO SCTO \
    --strategy scenario_based

# 전략 비교
python record_ensemble_gif.py \
    --models models/td3_tsco_map_500k.zip models/sac_stage4_final.zip \
    --compare-strategies \
    --map CSTO

# Q-value 전략으로 GIF 녹화
python record_ensemble_gif.py \
    --models models/td3_tsco_map_500k.zip models/sac_stage4_final.zip \
    --strategy q_value_weighted \
    --map OSCT \
    --seed 1000
```

---

## ⚙️ 시나리오별 가중치 커스터마이징

`evaluate_ensemble.py` 파일 수정:

```python
scenario_weights = {
    "CSTO": {"TD3": 0.85, "SAC": 0.15},  # TD3 가중치 증가
    "OSCT": {"TD3": 0.25, "SAC": 0.75},  # SAC 가중치 증가
    "TSCO": {"TD3": 0.6, "SAC": 0.4},
    "SCTO": {"TD3": 0.5, "SAC": 0.5},
    "default": {"TD3": 0.5, "SAC": 0.5}
}
```

---

## 🔧 문제 해결

### Q1: GIF가 생성되지 않습니다
- `--record-gif` 플래그를 추가했는지 확인
- 디렉토리 권한 확인
- 환경에 렌더링 지원 확인

### Q2: 모델 로드 실패
- 모델 경로가 올바른지 확인
- 모델 파일이 존재하는지 확인: `ls -la models/`

### Q3: 평가가 너무 느립니다
- `--episodes` 수를 줄이기 (예: 5로)
- `--maps` 수를 줄이기
- `q_value_weighted` 대신 `averaging` 사용 (Q-value 계산이 추가 시간 소요)

### Q4: Q-value 전략과 scenario_based 중 어떤 것을 선택?
**scenario_based** 추천:
- 각 맵별 모델 성능을 이미 알고 있을 때
- 최고 성능이 필요할 때
- 빠른 추론이 필요할 때

**q_value_weighted** 추천:
- 새로운 환경/맵에서 테스트할 때
- 모델별 최적 가중치를 모를 때
- 동적으로 변하는 환경에서 적응이 필요할 때

---

## 📚 더 알아보기

- 📖 [상세 가이드](ENSEMBLE_GUIDE.md) - 모든 기능과 예제
- 💻 [ensemble_agent.py](ensemble_agent.py) - 앙상블 에이전트 구현
- 🔬 [evaluate_ensemble.py](evaluate_ensemble.py) - 평가 스크립트
- 📊 [visualize_ensemble.py](visualize_ensemble.py) - 시각화 스크립트
- 🎬 [record_ensemble_gif.py](record_ensemble_gif.py) - GIF 녹화 스크립트

---

## 🎯 완전한 워크플로우 예제

```bash
# 1단계: 평가 실행
python evaluate_ensemble.py \
    --models models/td3_tsco_map_500k.zip models/sac_stage4_final.zip \
    --strategy scenario_based \
    --maps CSTO OSCT TSCO SCTO \
    --episodes 10 \
    --record-gif \
    --save ensemble_results.json

# 2단계: 결과 시각화
python visualize_ensemble.py \
    --results results/ensemble_results.json

# 3단계: 결과 확인
ls results/  # JSON, TXT, PNG 파일들 확인
ls gifs/ensemble/  # GIF 파일들 확인
```

---

**Happy Ensembling! 🚗💨**
