import os
from stable_baselines3 import PPO
from src.envs.space_invaders_env import SpaceInvadersEnv

def train():
    log_dir = "./logs"
    os.makedirs(log_dir, exist_ok=True)

    # 1. Среда без рендера
    env = SpaceInvadersEnv(render_mode=None)

    # 2. Передаем tensorboard_log напрямую
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,  # Поставим 1, чтобы в консоли хотя бы писало прогресс шагов
        tensorboard_log=log_dir
    )

    print("🚀 Старт обучения. Терминал должен быть ЧИСТЫМ от графики игры!")
    model.learn(total_timesteps=500000)

    model.save("./models/ppo_space_invaders_final")
    env.close()

if __name__ == "__main__":
    train()