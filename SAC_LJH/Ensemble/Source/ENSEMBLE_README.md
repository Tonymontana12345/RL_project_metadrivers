# 🤝 앙상블 모델 사용 가이드

TD3와 SAC 모델을 앙상블로 결합하여 더 나은 성능을 얻을 수 있습니다!

## 🚀 빠른 시작

### 1. 평가 + GIF 녹화 (한번에!)

```bash
python evaluate_ensemble.py \
    --models models/td3_tsco_map_500k.zip models/sac_tsco_map_500k.zip \
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

### 2. GIF만 빠르게 녹화

특정 맵에서 앙상블 주행을 빠르게 확인:

```bash
python record_ensemble_gif.py \
    --models models/td3_tsco_map_500k.zip models/sac_tsco_map_500k.zip \
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
    --models models/td3_tsco_map_500k.zip models/sac_tsco_map_500k.zip \
    --compare-strategies \
    --map CSTO \
    --seed 1000
```

**결과물:**
- `gifs/ensemble/comparison/averaging_CSTO_seed1000.gif`
- `gifs/ensemble/comparison/q_value_weighted_CSTO_seed1000.gif`
- `gifs/ensemble/comparison/scenario_based_CSTO_seed1000.gif`

---

## 📊 앙상블 전략

| 전략 | 설명 | 추천 |
|------|------|------|
| **scenario_based** ⭐ | 맵별로 각 모델에 다른 가중치 부여 | ✅ 권장! |
| **averaging** | 모든 모델의 액션을 평균 | ✅ 안정적 |
| **q_value_weighted** | Q-value에 비례하여 가중치 부여 | 동적 조정 |
| **confidence_weighted** | 불확실성 기반 가중치 | 고급 |
| **voting** | 투표 방식 (연속 액션에서는 평균과 유사) | 기본 |

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
python evaluate_ensemble.py \
    --models model1.zip model2.zip \
    --strategy scenario_based \
    --record-gif                    # 🎬 GIF 녹화 활성화
    --gif-dir gifs/my_ensemble      # 저장 위치 (선택)
```



### 수동 녹화

더 많은 제어가 필요하면 `record_ensemble_gif.py` 사용:

```bash
# 여러 시나리오 한번에
python record_ensemble_gif.py \
    --models model1.zip model2.zip \
    --maps CSTO OSCT TSCO SCTO \
    --strategy scenario_based

# 전략 비교
python record_ensemble_gif.py \
    --models model1.zip model2.zip \
    --compare-strategies \
    --map CSTO
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
- `q_value_weighted` 대신 `averaging` 사용

---

## 📚 더 알아보기

- 📖 [상세 가이드](ENSEMBLE_GUIDE.md) - 모든 기능과 예제
- 💻 [ensemble_agent.py](ensemble_agent.py) - 소스 코드
- 🔬 [evaluate_ensemble.py](evaluate_ensemble.py) - 평가 스크립트

---

**Happy Ensembling! 🚗💨**
