import os
from stable_baselines3 import PPO
from src.envs.space_invaders_env import SpaceInvadersEnv


def train():
    env = SpaceInvadersEnv(render_mode=None)

    model = PPO(
        "MlpPolicy",
        env,
        ent_coef=0.01,  # Принуждает модель пробовать разные действия и не застревать
        learning_rate=3e-4,
        verbose=1,
        tensorboard_log="./logs"
    )

    print("🚀 Старт обучения. Нажми Ctrl+C в любой момент, чтобы сохранить модель.")

    try:
        model.learn(total_timesteps=1000000)
    except KeyboardInterrupt:
        print("\n⚠️ Обучение прервано вручную!")

    # Этот блок сработает ВСЕГДА (и при завершении, и при Ctrl+C)
    os.makedirs("./models", exist_ok=True)
    save_path = "./models/ppo_space_invaders_final"
    model.save(save_path)

    print(f"✅ Модель успешно сохранена в {save_path}.zip!")
    env.close()


if __name__ == "__main__":
    train()