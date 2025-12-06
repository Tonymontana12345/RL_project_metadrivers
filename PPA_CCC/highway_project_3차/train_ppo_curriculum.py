"""
새로운 PPO 5단계 커리큘럼 러닝 학습 스크립트

실행 방법:
    python train_ppo_curriculum.py
"""

import os
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback

# 새로 만든 5단계 커리큘럼 설정 import
from config_ppo_curriculum import (
    BASE_ENV_CONFIG,
    PPO_CONFIG,
    CURRICULUM_STAGES,
    TRAINING_CONFIG,
    MODELS_DIR,
    LOGS_DIR,
    print_ppo_curriculum_info,
)

# 환경 생성 함수 및 시드 고정 함수 import
from envs.metadrive_env import make_env
from config import set_global_seed

def create_stage_env(stage_key, stage_config, seed=0):
    """
    커리큘럼 단계에 맞는 환경을 생성합니다.
    
    Args:
        stage_key (str): 현재 단계의 키 (예: "stage_1")
        stage_config (dict): 현재 단계의 설정값
        seed (int): 환경 생성에 사용할 시드
    
    Returns:
        DummyVecEnv: 벡터화된 환경
    """
    env_config = BASE_ENV_CONFIG.copy()
    
    # 단계별 decision_repeat 값 동적 설정
    if stage_key in ["stage_4", "stage_5"]:
        env_config["decision_repeat"] = 2
    else:
        env_config["decision_repeat"] = 1
        
    # 단계별 설정 적용 (map, traffic_density, random_traffic)
    env_config.update({
        "map": stage_config["map"],
        "traffic_density": stage_config["traffic_density"],
        "random_traffic": stage_config["random_traffic"],
        "start_seed": seed,
        "num_scenarios": 1000, # 각 환경은 1000개의 다양한 시나리오로 구성
    })
    
    # 벡터화된 환경 생성
    vec_env = DummyVecEnv([make_env(seed=seed, render=False, config=env_config)])
    return vec_env

def train_ppo_curriculum():
    """
    새로운 5단계 PPO 커리큘럼 러닝을 수행합니다.
    """
    # 커리큘럼 정보 출력
    print_ppo_curriculum_info()
    
    # 시드 고정 (재현성을 위해)
    set_global_seed(42)
    
    model_name = TRAINING_CONFIG['model_name']
    log_dir = os.path.join(LOGS_DIR, model_name)
    os.makedirs(log_dir, exist_ok=True)
    
    print("\n" + "="*70)
    print("🚀 PPO 5-Stage Curriculum Learning Started!")
    print(f"Model will be saved with prefix: {model_name}")
    print(f"TensorBoard Log Directory: {log_dir}")
    print("="*70 + "\n")
    
    model = None
    total_steps_trained = 0
    
    # 5단계 커리큘럼 학습 루프
    for i, (stage_key, stage_config) in enumerate(CURRICULUM_STAGES.items()):
        stage_num = i + 1
        
        print("\n" + "="*70)
        print(f"📚 Stage {stage_num}/{len(CURRICULUM_STAGES)}: {stage_config['name']}")
        print("="*70)
        
        # 1. 환경 생성
        print("🌍 Creating environment...")
        # 각 스테이지마다 겹치지 않는 1000개의 시드 풀을 사용
        stage_seed = 1000 * (i + 1)
        env = create_stage_env(stage_key, stage_config, seed=stage_seed)
        print(f"✅ Environment created (Map: '{stage_config['map']}', Traffic: {stage_config['traffic_density']})")
        print(f"   (Seed Pool: {stage_seed} ~ {stage_seed + 999})")
        
        # 2. 모델 생성 또는 로드
        if model is None:
            print("\n🆕 Creating a new PPO model...")
            # PPO_CONFIG에 tensorboard_log 경로 추가
            ppo_params = PPO_CONFIG.copy()
            ppo_params["tensorboard_log"] = log_dir
            model = PPO("MlpPolicy", env, **ppo_params)
            print("✅ New model created.")
        else:
            print(f"\n🔄 Loading model from Stage {stage_num-1} and setting new environment...")
            model.set_env(env)
            print("✅ Environment updated for the new stage.")
            
        # 3. 콜백 설정 (체크포인트 저장용)
        checkpoint_callback = CheckpointCallback(
            save_freq=TRAINING_CONFIG['save_freq'],
            save_path=MODELS_DIR,
            name_prefix=f"{model_name}_{stage_key}",
            save_replay_buffer=False,
            save_vecnormalize=True,
        )
        
        # 4. 학습 시작
        print(f"\n🎓 Starting training for Stage {stage_num}...")
        try:
            model.learn(
                total_timesteps=stage_config["steps"],
                callback=checkpoint_callback,
                progress_bar=True,
                reset_num_timesteps=False, # 이전 단계의 스텝 수를 이어서 카운트
            )
        except KeyboardInterrupt:
            print("\n\n⚠️ Training interrupted by user. Saving current model...")
            break # 루프를 중단하고 최종 모델 저장으로 이동
            
        total_steps_trained += model.num_timesteps
        
        # 5. 단계별 모델 저장
        stage_model_path = os.path.join(MODELS_DIR, f"{model_name}_{stage_key}.zip")
        model.save(stage_model_path)
        
        print("\n" + "-"*70)
        print(f"💾 Stage {stage_num} finished! Model saved to: {stage_model_path}")
        print(f"Total timesteps so far: {model.num_timesteps:,}")
        print("-" * 70)
        
        env.close()

    # 5. 최종 모델 저장
    final_model_path = os.path.join(MODELS_DIR, f"{model_name}_final.zip")
    model.save(final_model_path)
    
    print("\n" + "="*70)
    print("🎉🎉 PPO 5-Stage Curriculum Learning Finished! 🎉🎉")
    print(f"Final model saved to: {final_model_path}")
    print(f"Total timesteps trained: {model.num_timesteps:,}")
    print("\nTo evaluate the final model, you can run:")
    print(f"python evaluate.py --model {final_model_path}")
    print("="*70 + "\n")

if __name__ == "__main__":
    train_ppo_curriculum()
