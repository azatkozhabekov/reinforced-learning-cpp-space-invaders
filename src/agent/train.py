import os
import torch as th
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack
from src.envs.space_invaders_env import SpaceInvadersEnv


# 💥 Компактная CNN специально под низкое разрешение (20x30)
class SmallGridCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)

        n_input_channels = observation_space.shape[0]

        self.cnn = nn.Sequential(
            # Слой 1: Вход (n_channels, 20, 30) -> Выход (32, 18, 28)
            nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            # Слой 2: Выход (64, 16, 26)
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            # Слой 3: Выход (64, 7, 12)
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Вычисляем итоговый размер вектора признаков после сверток
        with th.no_grad():
            sample_input = th.as_tensor(observation_space.sample()[None]).float()
            n_flatten = self.cnn(sample_input).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))


def make_env():
    def _init():
        return SpaceInvadersEnv(render_mode=None)

    return _init


def train():
    num_envs = 16
    print(f"🔥 Создание {num_envs} параллельных процессов C++...")

    env = SubprocVecEnv([make_env() for _ in range(num_envs)])
    env = VecFrameStack(env, n_stack=4, channels_order="first")

    # Передаем нашу кастомную архитектуру сети через policy_kwargs
    policy_kwargs = dict(
        features_extractor_class=SmallGridCNN,
        features_extractor_kwargs=dict(features_dim=256),
        normalize_images=False,
    )

    model = PPO(
        "CnnPolicy",
        env,
        verbose=1,
        learning_rate=0.0003,
        n_steps=128,
        batch_size=64,
        ent_coef=0.05,
        policy_kwargs=policy_kwargs,  # 💥 Подключаем компактную CNN
        tensorboard_log="./logs"
    )

    print("🚀 Старт обучения SmallGridCNN + PPO...")
    try:
        model.learn(total_timesteps=1000000)
    except KeyboardInterrupt:
        print("\n⚠️ Обучение прервано вручную!")
    finally:
        os.makedirs("./models", exist_ok=True)
        model.save("./models/ppo_space_invaders_final")
        env.close()


if __name__ == "__main__":
    train()