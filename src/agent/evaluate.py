import time
from stable_baselines3 import PPO
from src.envs.space_invaders_env import SpaceInvadersEnv


def evaluate():
    # 1. Создаем среду С ВКЛЮЧЕННЫМ визуальным режимом
    env = SpaceInvadersEnv(render_mode="human")

    # 2. Загружаем сохранённую модель из файла .zip
    model_path = "./models/ppo_space_invaders_final.zip"
    print(f"📂 Загрузка обученной модели из {model_path}...")
    model = PPO.load(model_path, env=env)

    # 3. Смотрим на работу умного агента
    obs, _ = env.reset()
    done = False
    total_reward = 0

    print("👾 Запуск тестовой игры!")
    while not done:
        # deterministic=True заставляет модель выбирать НАИЛУЧШЕЕ выученное действие
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        total_reward += reward
        done = terminated or truncated

    print(f"🎮 Игра окончена! Финальный счёт: {info.get('score', 0)}")
    env.close()


if __name__ == "__main__":
    evaluate()