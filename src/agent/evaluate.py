import os
import time
from stable_baselines3 import PPO
from src.envs.space_invaders_env import SpaceInvadersEnv
import config


def test_best_model():
    model_path = os.path.join(config.MODELS_DIR, "ppo_space_invaders_final")

    if not os.path.exists(model_path + ".zip"):
        print("Обученная модель не найдена! Сначала запустите train.py")
        return

    env = SpaceInvadersEnv()
    model = PPO.load(model_path)

    obs, _ = env.reset()
    done = False
    total_score = 0

    print("Запуск демонстрационного прогона...")
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, info = env.step(action)
        total_score = info.get("score", 0)
        time.sleep(0.05)  # Задержка для визуального контроля в терминале

    print(f"Попытка завершена! Итоговый счёт: {total_score}")


if __name__ == "__main__":
    test_best_model()