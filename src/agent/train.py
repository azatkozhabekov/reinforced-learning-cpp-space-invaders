import os
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from src.envs.space_invaders_env import SpaceInvadersEnv
import config


def train():
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    os.makedirs(config.LOGS_DIR, exist_ok=True)

    env = SpaceInvadersEnv()

    # Модель PPO, заточенная под обучение на максимизацию награды
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=0.0003,
        tensorboard_log=config.LOGS_DIR
    )

    # Сохраняем чеклисты каждые 10 000 шагов
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=config.MODELS_DIR,
        name_prefix="ppo_space_invaders"
    )

    print("Запуск процесса обучения...")
    model.learn(total_timesteps=100000, callback=checkpoint_callback)

    model.save(os.path.join(config.MODELS_DIR, "ppo_space_invaders_final"))
    print("Обучение завершено. Модель сохранена в models/")


if __name__ == "__main__":
    train()