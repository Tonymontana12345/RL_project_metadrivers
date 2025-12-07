"""
앙상블 에이전트 모듈

TDS(TD3)와 SAC 모델을 결합하여 여러 앙상블 전략을 제공합니다.
"""

import numpy as np
import torch
from stable_baselines3 import TD3, SAC
from typing import List, Dict, Tuple, Optional, Literal


class EnsembleAgent:
    """
    여러 RL 모델을 앙상블하는 에이전트

    지원하는 앙상블 전략:
    1. voting: 액션 투표 (다수결)
    2. averaging: 액션 평균
    3. q_value_weighted: Q-value 기반 가중 평균
    4. scenario_based: 시나리오별 모델 선택
    5. confidence_weighted: 신뢰도 기반 가중 평균
    """

    def __init__(
        self,
        models: List,
        model_names: List[str] = None,
        strategy: Literal["voting", "averaging", "q_value_weighted",
                         "scenario_based", "confidence_weighted"] = "averaging",
        scenario_weights: Optional[Dict[str, Dict[str, float]]] = None
    ):
        """
        Args:
            models: 앙상블할 모델 리스트 (예: [td3_model, sac_model])
            model_names: 모델 이름 리스트 (예: ["TD3", "SAC"])
            strategy: 앙상블 전략
            scenario_weights: 시나리오별 모델 가중치 (scenario_based 전략용)
        """
        self.models = models
        self.model_names = model_names or [f"Model_{i}" for i in range(len(models))]
        self.strategy = strategy
        self.scenario_weights = scenario_weights or {}

        print(f"🤝 앙상블 에이전트 초기화")
        print(f"   - 모델 수: {len(self.models)}")
        print(f"   - 모델 이름: {self.model_names}")
        print(f"   - 전략: {self.strategy}")

    def predict(
        self,
        observation,
        deterministic: bool = True,
        scenario: Optional[str] = None
    ) -> Tuple[np.ndarray, Optional[Dict]]:
        """
        앙상블 예측

        Args:
            observation: 환경 관찰
            deterministic: 결정적 예측 여부
            scenario: 시나리오 타입 (scenario_based 전략용)

        Returns:
            (action, info_dict)
        """
        if self.strategy == "voting":
            return self._voting_predict(observation, deterministic)
        elif self.strategy == "averaging":
            return self._averaging_predict(observation, deterministic)
        elif self.strategy == "q_value_weighted":
            return self._q_value_weighted_predict(observation, deterministic)
        elif self.strategy == "scenario_based":
            return self._scenario_based_predict(observation, deterministic, scenario)
        elif self.strategy == "confidence_weighted":
            return self._confidence_weighted_predict(observation, deterministic)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

    def _voting_predict(self, observation, deterministic: bool) -> Tuple[np.ndarray, Dict]:
        """투표 방식: 각 모델의 액션 중 가장 가까운 액션들의 평균"""
        actions = []

        for model in self.models:
            action, _ = model.predict(observation, deterministic=deterministic)
            actions.append(action)

        # 연속 액션 공간이므로 평균 사용 (이산 액션이면 최빈값 사용)
        final_action = np.mean(actions, axis=0)

        info = {
            "strategy": "voting",
            "individual_actions": actions,
            "final_action": final_action
        }

        return final_action, info

    def _averaging_predict(self, observation, deterministic: bool) -> Tuple[np.ndarray, Dict]:
        """평균 방식: 모든 모델 액션의 단순 평균"""
        actions = []

        for model in self.models:
            action, _ = model.predict(observation, deterministic=deterministic)
            actions.append(action)

        final_action = np.mean(actions, axis=0)

        info = {
            "strategy": "averaging",
            "individual_actions": actions,
            "final_action": final_action
        }

        return final_action, info

    def _q_value_weighted_predict(self, observation, deterministic: bool) -> Tuple[np.ndarray, Dict]:
        """Q-value 기반 가중 평균: 각 모델의 Q-value에 비례하여 가중치 부여"""
        actions = []
        q_values = []

        for model in self.models:
            action, _ = model.predict(observation, deterministic=deterministic)
            actions.append(action)

            # Q-value 추정
            try:
                obs_tensor = model.policy.obs_to_tensor(observation)[0]

                if hasattr(model, 'critic') and hasattr(model.policy, 'scale_action'):
                    # SAC의 경우
                    action_tensor = model.policy.scale_action(action)
                    with torch.no_grad():
                        q_value = model.critic(obs_tensor, action_tensor)[0].mean().item()
                    q_values.append(q_value)
                elif hasattr(model.policy, 'actor') and hasattr(model, 'critic'):
                    # TD3의 경우
                    with torch.no_grad():
                        action_tensor = model.policy.actor(obs_tensor)
                        q_value = model.critic(obs_tensor, action_tensor)[0].mean().item()
                    q_values.append(q_value)
                else:
                    # Q-value를 추정할 수 없으면 동일 가중치
                    q_values.append(1.0)
            except Exception as e:
                # 에러 발생 시 기본 가중치 사용
                q_values.append(1.0)

        # Q-value를 확률로 정규화 (softmax)
        q_values = np.array(q_values)
        exp_q = np.exp(q_values - np.max(q_values))  # 수치 안정성
        weights = exp_q / np.sum(exp_q)

        # 가중 평균
        actions = np.array(actions)
        final_action = np.sum(actions * weights[:, np.newaxis], axis=0)

        info = {
            "strategy": "q_value_weighted",
            "individual_actions": actions.tolist(),
            "q_values": q_values.tolist(),
            "weights": weights.tolist(),
            "final_action": final_action
        }

        return final_action, info

    def _scenario_based_predict(
        self,
        observation,
        deterministic: bool,
        scenario: Optional[str]
    ) -> Tuple[np.ndarray, Dict]:
        """
        시나리오 기반 선택: 시나리오에 따라 모델별 가중치 조정

        예시:
        scenario_weights = {
            "CSTO": {"TD3": 0.8, "SAC": 0.2},  # TD3가 CSTO에서 강함
            "OSCT": {"TD3": 0.3, "SAC": 0.7},  # SAC가 OSCT에서 강함
            "default": {"TD3": 0.5, "SAC": 0.5}
        }
        """
        actions = []

        for model in self.models:
            action, _ = model.predict(observation, deterministic=deterministic)
            actions.append(action)

        # 시나리오별 가중치 가져오기
        if scenario and scenario in self.scenario_weights:
            weights_dict = self.scenario_weights[scenario]
        elif "default" in self.scenario_weights:
            weights_dict = self.scenario_weights["default"]
        else:
            # 기본값: 균등 가중치
            weights_dict = {name: 1.0/len(self.models) for name in self.model_names}

        # 모델 이름에 맞춰 가중치 배열 생성
        weights = np.array([weights_dict.get(name, 1.0/len(self.models))
                           for name in self.model_names])
        weights = weights / np.sum(weights)  # 정규화

        # 가중 평균
        actions = np.array(actions)
        final_action = np.sum(actions * weights[:, np.newaxis], axis=0)

        info = {
            "strategy": "scenario_based",
            "scenario": scenario,
            "individual_actions": actions.tolist(),
            "weights": weights.tolist(),
            "final_action": final_action
        }

        return final_action, info

    def _confidence_weighted_predict(self, observation, deterministic: bool) -> Tuple[np.ndarray, Dict]:
        """
        신뢰도 기반 가중 평균: 각 모델의 행동 분산(불확실성)을 기반으로 가중치 부여
        분산이 작을수록(확신이 높을수록) 가중치가 높음
        """
        actions = []
        uncertainties = []

        for model in self.models:
            # 여러 샘플링으로 불확실성 추정
            if not deterministic:
                action, _ = model.predict(observation, deterministic=False)
            else:
                # deterministic일 때는 여러 번 샘플링
                sample_actions = []
                for _ in range(5):
                    sample_action, _ = model.predict(observation, deterministic=False)
                    sample_actions.append(sample_action)

                # 평균과 표준편차
                action = np.mean(sample_actions, axis=0)
                uncertainty = np.mean(np.std(sample_actions, axis=0))
                uncertainties.append(uncertainty)

            actions.append(action)

        # 불확실성이 없으면 기본값
        if not uncertainties or all(u == 0 for u in uncertainties):
            uncertainties = [1.0] * len(self.models)

        # 불확실성의 역수를 가중치로 사용 (낮은 불확실성 = 높은 가중치)
        uncertainties = np.array(uncertainties)
        weights = 1.0 / (uncertainties + 1e-8)  # 0으로 나누기 방지
        weights = weights / np.sum(weights)

        # 가중 평균
        actions = np.array(actions)
        final_action = np.sum(actions * weights[:, np.newaxis], axis=0)

        info = {
            "strategy": "confidence_weighted",
            "individual_actions": actions.tolist(),
            "uncertainties": uncertainties.tolist(),
            "weights": weights.tolist(),
            "final_action": final_action
        }

        return final_action, info


def load_ensemble_models(model_paths: List[str], algorithm_types: List[str] = None):
    """
    여러 모델 파일을 로드하여 앙상블용 모델 리스트 생성

    Args:
        model_paths: 모델 파일 경로 리스트
        algorithm_types: 각 모델의 알고리즘 타입 (None이면 자동 감지)

    Returns:
        (models, model_names)
    """
    from evaluate import load_model, detect_algorithm

    models = []
    model_names = []

    for i, path in enumerate(model_paths):
        if algorithm_types and i < len(algorithm_types):
            algo = algorithm_types[i]
        else:
            algo = detect_algorithm(path)

        model = load_model(path, algorithm=algo)
        models.append(model)
        model_names.append(algo.upper())

        print(f"✅ 모델 로드 완료: {path} ({algo.upper()})")

    return models, model_names


# ============================================================
# 사용 예시
# ============================================================

if __name__ == "__main__":
    """
    앙상블 에이전트 테스트

    사용법:
        python ensemble_agent.py
    """
    import gym
    from metadrive import MetaDriveEnv
    from config import FIXED_SEED_ENV_CONFIG

    print("=" * 60)
    print("앙상블 에이전트 테스트")
    print("=" * 60)

    # 예시: 두 모델 경로
    model_paths = [
        "models/td3_tsco_map_500k.zip",
        "models/sac_stage4_final.zip"
    ]

    # 모델 로드
    print("\n📦 모델 로드 중...")
    models, model_names = load_ensemble_models(model_paths)

    # 시나리오별 가중치 예시
    scenario_weights = {
        "CSTO": {"TD3": 0.8, "SAC": 0.2},  # TD3가 CSTO에서 강함
        "OSCT": {"TD3": 0.3, "SAC": 0.7},  # SAC가 OSCT에서 강함
        "TSCO": {"TD3": 0.5, "SAC": 0.5},
        "default": {"TD3": 0.5, "SAC": 0.5}
    }

    # 앙상블 에이전트 생성 (여러 전략 테스트)
    strategies = ["averaging", "q_value_weighted", "scenario_based"]

    for strategy in strategies:
        print(f"\n{'=' * 60}")
        print(f"전략: {strategy}")
        print(f"{'=' * 60}")

        ensemble = EnsembleAgent(
            models=models,
            model_names=model_names,
            strategy=strategy,
            scenario_weights=scenario_weights
        )

        # 환경 생성
        import copy
        env_config = copy.deepcopy(FIXED_SEED_ENV_CONFIG)
        env_config["map"] = "CSTO"

        # 센서 설정을 모델 학습 시와 동일하게 (관찰 공간 91 차원)
        # 기존 vehicle_config를 완전히 덮어쓰기
        env_config["vehicle_config"] = {
            "lidar": {
                "num_lasers": 72,
                "distance": 70,
            }
        }

        env = MetaDriveEnv(env_config)

        # 테스트 실행
        obs, _ = env.reset()
        total_reward = 0
        done = False
        steps = 0

        print(f"\n🚗 시나리오 CSTO에서 테스트 실행 중...")

        while not done and steps < 100:
            # scenario_based인 경우 시나리오 전달
            if strategy == "scenario_based":
                action, info = ensemble.predict(obs, deterministic=True, scenario="CSTO")
            else:
                action, info = ensemble.predict(obs, deterministic=True)

            obs, reward, terminated, truncated, info_env = env.step(action)
            done = terminated or truncated
            total_reward += reward
            steps += 1

        print(f"   - 총 스텝: {steps}")
        print(f"   - 총 보상: {total_reward:.2f}")
        print(f"   - 성공: {info_env.get('arrive_dest', False)}")

        env.close()

    print(f"\n{'=' * 60}")
    print("✅ 테스트 완료")
    print(f"{'=' * 60}")
