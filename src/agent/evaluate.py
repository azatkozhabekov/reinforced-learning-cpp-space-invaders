import time
from stable_baselines3 import PPO
from src.envs.space_invaders_env import SpaceInvadersEnv


def evaluate():
    model_path = "./models/ppo_space_invaders_final"

    print(f"📂 Загрузка обученной модели из {model_path}.zip...")

    # 1. Инициализируем среду с визуальным режимом "human"
    env = SpaceInvadersEnv(render_mode="human")
    model = PPO.load(model_path, env=env)

    print("👾 Запуск тестовой игры! Нажмите Ctrl+C для выхода.")

    obs, _ = env.reset()
    try:
        while True:
            # Модель выбирает детерминированное (лучшее) действие
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

            # Регулировка FPS во время просмотра (20 FPS = 0.05 сек задержки)
            time.sleep(0.05)

            if terminated or truncated:
                print("🎮 Игра окончена! Перезапуск...")
                obs, _ = env.reset()
    except KeyboardInterrupt:
        print("\n👋 Просмотр завершён.")
    finally:
        env.close()


if __name__ == "__main__":
    evaluate()