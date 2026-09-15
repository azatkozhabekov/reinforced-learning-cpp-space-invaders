import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from src.envs.space_invaders_env import SpaceInvadersEnv


def make_env():
    """Фабрика для создания независимых экземпляров среды."""

    def _init():
        return SpaceInvadersEnv(render_mode=None)

    return _init


def train():
    # Указываем количество параллельных процессов C++ (8, 16 или 32 в зависимости от ЦП)
    num_envs = 16

    print(f"🔥 Создание {num_envs} параллельных процессов C++...")
    env = SubprocVecEnv([make_env() for _ in range(num_envs)])

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=0.0003,
        ent_coef=0.05,
        n_steps=128,  # 128 шагов * 16 процессов = 2048 шагов за итерацию!
        batch_size=64,
        tensorboard_log="./logs"
    )

    print("🚀 Старт параллельного обучения. Нажми Ctrl+C в любой момент для сохранения.")

    try:
        model.learn(total_timesteps=1000000)
    except KeyboardInterrupt:
        print("\n⚠️ Обучение прервано вручную!")
    finally:
        os.makedirs("./models", exist_ok=True)
        save_path = "./models/ppo_space_invaders_final"
        model.save(save_path)
        print(f"✅ Модель успешно сохранена в {save_path}.zip!")
        env.close()


if __name__ == "__main__":
    train()