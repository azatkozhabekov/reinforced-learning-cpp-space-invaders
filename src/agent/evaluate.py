import time
from stable_baselines3 import PPO
from src.envs.space_invaders_env import SpaceInvadersEnv


def evaluate():
    model_path = "./models/ppo_space_invaders_final"

    print(f"📂 Загрузка обученной модели из {model_path}.zip...")

    # 1. Запускаем среду с выводом в консоль
    env = SpaceInvadersEnv(render_mode="human")
    model = PPO.load(model_path, env=env)

    print("👾 Запуск тестовой игры! Нажмите Ctrl+C для выхода.\n")

    obs, _ = env.reset()
    try:
        while True:
            # Модель выбирает наилучшее действие
            action, _ = model.predict(obs, deterministic=True)

            # Передаем действие в C++
            obs, reward, terminated, truncated, info = env.step(action)

            if terminated or truncated:
                print("\n🎮 Раунд окончен! Перезапуск...\n")
                obs, _ = env.reset()
    except KeyboardInterrupt:
        print("\n👋 Просмотр завершён.")
    finally:
        env.close()


if __name__ == "__main__":
    evaluate()