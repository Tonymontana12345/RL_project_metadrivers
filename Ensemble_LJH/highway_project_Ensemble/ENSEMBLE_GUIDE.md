# 앙상블 에이전트 사용 가이드

TD3와 SAC 모델을 결합하여 더 나은 성능을 얻는 앙상블 기법입니다.

## 📋 목차
1. [앙상블 전략 설명](#앙상블-전략-설명)
2. [빠른 시작](#빠른-시작)
3. [전략별 사용 예시](#전략별-사용-예시)
4. [성능 비교](#성능-비교)

---

## 앙상블 전략 설명

### 1. **averaging** (평균 전략) - 가장 안정적
- 모든 모델의 액션을 단순 평균
- 장점: 구현이 간단하고 안정적
- 단점: 각 모델의 강점을 충분히 활용하지 못할 수 있음

### 2. **q_value_weighted** (Q-value 가중 전략) - 동적 가중
- 각 모델의 Q-value에 비례하여 가중치 부여
- 장점: 현재 상황에서 더 확신이 높은 모델의 의견을 더 반영
- 단점: Q-value 계산 오버헤드 발생

### 3. **scenario_based** (시나리오 기반 전략) - 권장 ⭐
- 시나리오(맵)별로 미리 정의된 가중치 사용
- 장점: TD3는 CSTO에서, SAC는 OSCT에서 강점을 살림
- 사용 예시:
  ```python
  scenario_weights = {
      "CSTO": {"TD3": 0.8, "SAC": 0.2},  # TD3가 CSTO에서 강함
      "OSCT": {"TD3": 0.3, "SAC": 0.7},  # SAC가 OSCT에서 그나마 나음
      "TSCO": {"TD3": 0.5, "SAC": 0.5},
      "default": {"TD3": 0.5, "SAC": 0.5}
  }
  ```

### 4. **confidence_weighted** (신뢰도 기반 전략)
- 각 모델의 행동 분산(불확실성)을 기반으로 가중치 부여
- 장점: 불확실성이 낮은 모델의 의견을 더 반영
- 단점: 여러 번 샘플링해야 하므로 느림

### 5. **voting** (투표 전략)
- 각 모델의 액션 투표 (연속 공간에서는 평균과 유사)
- 장점: 이해하기 쉬움
- 단점: 연속 액션 공간에서는 averaging과 큰 차이 없음

---

## 빠른 시작

### 설치 확인
```bash
# 필요한 파일이 있는지 확인
ls ensemble_agent.py
ls evaluate_ensemble.py
ls record_ensemble_gif.py
```

### 기본 실행 (평균 전략)
```bash
python evaluate_ensemble.py \
    --models models/td3_tsco_map_500k.zip models/sac_tsco_map_500k.zip \
    --strategy averaging \
    --episodes 10
```

### 시나리오 기반 전략 (권장) + GIF 녹화
```bash
python evaluate_ensemble.py \
    --models models/td3_tsco_map_500k.zip models/sac_tsco_map_500k.zip \
    --strategy scenario_based \
    --maps CSTO OSCT TSCO SCTO \
    --episodes 10 \
    --save ensemble_scenario_based.json \
    --record-gif
```

### Q-value 가중 전략
```bash
python evaluate_ensemble.py \
    --models models/td3_tsco_map_500k.zip models/sac_tsco_map_500k.zip \
    --strategy q_value_weighted \
    --episodes 10 \
    --save ensemble_qvalue.json
```

---

## 전략별 사용 예시

### 코드에서 직접 사용하기

```python
from ensemble_agent import EnsembleAgent, load_ensemble_models
from metadrive import MetaDriveEnv
from config import FIXED_SEED_ENV_CONFIG

# 1. 모델 로드
model_paths = [
    "models/td3_tsco_map_500k.zip",
    "models/sac_tsco_map_500k.zip"
]
models, model_names = load_ensemble_models(model_paths)

# 2. 시나리오별 가중치 정의
scenario_weights = {
    "CSTO": {"TD3": 0.8, "SAC": 0.2},
    "OSCT": {"TD3": 0.3, "SAC": 0.7},
    "default": {"TD3": 0.5, "SAC": 0.5}
}

# 3. 앙상블 에이전트 생성
ensemble = EnsembleAgent(
    models=models,
    model_names=model_names,
    strategy="scenario_based",
    scenario_weights=scenario_weights
)

# 4. 환경에서 실행
env_config = FIXED_SEED_ENV_CONFIG.copy()
env_config["map"] = "CSTO"
env = MetaDriveEnv(env_config)

obs = env.reset()
done = False
total_reward = 0

while not done:
    # 현재 맵(시나리오)을 전달
    action, info = ensemble.predict(obs, deterministic=True, scenario="CSTO")
    obs, reward, done, _ = env.step(action)
    total_reward += reward

print(f"Total Reward: {total_reward:.2f}")
env.close()
```

---

## 성능 비교

### 예상 결과

| 전략 | CSTO 성공률 | OSCT 성공률 | 전체 평균 | 특징 |
|------|-------------|-------------|-----------|------|
| TD3 단독 | ~80%+ | ~0% | ~40% | CSTO에서 강함 |
| SAC 단독 | ~0% | ~33% | ~16% | OSCT에서 그나마 나음 |
| **averaging** | ~40% | ~16% | ~28% | 안정적이지만 보수적 |
| **scenario_based** | ~64%+ | ~23%+ | ~43%+ | ⭐ 최고 성능 기대 |
| **q_value_weighted** | ~50% | ~20% | ~35% | 동적 조정 |

> **주의**: 실제 결과는 모델 품질과 하이퍼파라미터에 따라 다를 수 있습니다.

---

## 가중치 튜닝 팁

### CSTO 시나리오
- TD3가 강하므로 가중치를 높게 설정
- 예: `{"TD3": 0.7~0.9, "SAC": 0.1~0.3}`

### OSCT 시나리오
- SAC가 그나마 성능이 나오므로 가중치 증가
- 예: `{"TD3": 0.2~0.4, "SAC": 0.6~0.8}`

### 균형 잡힌 시나리오 (TSCO, SCTO)
- 동등한 가중치 또는 약간의 차이
- 예: `{"TD3": 0.5, "SAC": 0.5}`

### 가중치 최적화 방법
```python
# 그리드 서치로 최적 가중치 찾기
for td3_weight in [0.5, 0.6, 0.7, 0.8, 0.9]:
    sac_weight = 1.0 - td3_weight
    scenario_weights = {
        "CSTO": {"TD3": td3_weight, "SAC": sac_weight}
    }
    # 평가 실행...
```

---

## 문제 해결

### Q1: "모델을 로드할 수 없습니다" 오류
```bash
# 모델 파일 경로 확인
ls -la models/

# 절대 경로 사용
python evaluate_ensemble.py \
    --models /absolute/path/to/td3_model.zip /absolute/path/to/sac_model.zip
```

### Q2: Q-value 가중 전략이 느림
- 이는 정상입니다. Q-value 계산이 추가로 필요하기 때문입니다.
- 더 빠른 성능이 필요하면 `averaging` 또는 `scenario_based` 사용

### Q3: 시나리오별 가중치 수정하기
`evaluate_ensemble.py` 파일의 `scenario_weights` 딕셔너리를 수정:

```python
scenario_weights = {
    "CSTO": {"TD3": 0.85, "SAC": 0.15},  # 가중치 조정
    "OSCT": {"TD3": 0.25, "SAC": 0.75},
    # ...
}
```

---

## GIF 녹화 사용법

### 평가와 함께 GIF 녹화
각 맵의 첫 번째 에피소드만 GIF로 자동 녹화:

```bash
python evaluate_ensemble.py \
    --models models/td3_tsco_map_500k.zip models/sac_tsco_map_500k.zip \
    --strategy scenario_based \
    --maps CSTO OSCT \
    --episodes 5 \
    --record-gif \
    --gif-dir gifs/ensemble
```

생성되는 GIF:
- `gifs/ensemble/ensemble_scenario_based_CSTO_seed1000.gif`
- `gifs/ensemble/ensemble_scenario_based_OSCT_seed1000.gif`

### 독립적으로 GIF만 녹화

#### 단일 시나리오 녹화
```bash
python record_ensemble_gif.py \
    --models models/td3_tsco_map_500k.zip models/sac_tsco_map_500k.zip \
    --strategy scenario_based \
    --map CSTO \
    --seed 1000 \
    --output ensemble_csto.gif
```

#### 여러 시나리오 한번에 녹화
```bash
python record_ensemble_gif.py \
    --models models/td3_tsco_map_500k.zip models/sac_tsco_map_500k.zip \
    --strategy scenario_based \
    --maps CSTO OSCT TSCO SCTO \
    --seed 1000 \
    --output-dir gifs/ensemble
```

#### 전략 비교 녹화 (같은 맵에서 여러 전략)
```bash
python record_ensemble_gif.py \
    --models models/td3_tsco_map_500k.zip models/sac_tsco_map_500k.zip \
    --compare-strategies \
    --map CSTO \
    --seed 1000
```

생성되는 GIF:
- `gifs/ensemble/comparison/averaging_CSTO_seed1000.gif`
- `gifs/ensemble/comparison/q_value_weighted_CSTO_seed1000.gif`
- `gifs/ensemble/comparison/scenario_based_CSTO_seed1000.gif`

---

## 고급 사용법

### 3개 이상의 모델 앙상블
```python
model_paths = [
    "models/td3_model1.zip",
    "models/sac_model1.zip",
    "models/ppo_model1.zip"
]
models, model_names = load_ensemble_models(model_paths)

# 가중치도 3개로 확장
scenario_weights = {
    "CSTO": {"TD3": 0.5, "SAC": 0.3, "PPO": 0.2}
}
```

### 커스텀 앙상블 전략 구현
`ensemble_agent.py`에 새로운 메서드 추가:

```python
def _custom_predict(self, observation, deterministic):
    # 여기에 커스텀 로직 구현
    actions = [model.predict(observation, deterministic=deterministic)[0]
               for model in self.models]

    # 예: 첫 번째 모델에 70% 가중치
    final_action = 0.7 * actions[0] + 0.3 * actions[1]

    return final_action, {"strategy": "custom"}
```

---

## 추가 리소스

- [ensemble_agent.py](ensemble_agent.py) - 앙상블 에이전트 구현
- [evaluate_ensemble.py](evaluate_ensemble.py) - 평가 스크립트 (GIF 녹화 지원)
- [record_ensemble_gif.py](record_ensemble_gif.py) - GIF 녹화 전용 스크립트
- [config.py](config.py) - 환경 설정

## 빠른 참고 명령어

### 평가 + GIF 녹화 (한번에!)
```bash
# 시나리오 기반 전략으로 평가하고 GIF도 저장
python evaluate_ensemble.py \
    --models models/td3_tsco_map_500k.zip models/sac_tsco_map_500k.zip \
    --strategy scenario_based \
    --maps CSTO OSCT \
    --episodes 5 \
    --record-gif \
    --save ensemble_results.json
```

### GIF만 빠르게 녹화
```bash
# CSTO 맵만 빠르게 확인
python record_ensemble_gif.py \
    --models models/td3_tsco_map_500k.zip models/sac_tsco_map_500k.zip \
    --strategy scenario_based \
    --map CSTO \
    --seed 1000
```

### 전략 비교 시각화
```bash
# 여러 전략을 비교하는 GIF 생성
python record_ensemble_gif.py \
    --models models/td3_tsco_map_500k.zip models/sac_tsco_map_500k.zip \
    --compare-strategies \
    --map CSTO
```

---

**마지막 업데이트**: 2025-11-26
