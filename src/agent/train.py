import os
import torch as th
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack, VecMonitor
from src.envs.space_invaders_env import SpaceInvadersEnv

class SmallGridCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)

        n_input_channels = observation_space.shape[0]

        self.cnn = nn.Sequential(
            # layer 1: exit (n_channels, 20, 30) -> exit (32, 18, 28)
            nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            # layer 2: exit (64, 16, 26)
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            # layer 3: exit (64, 7, 12)
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

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
    print(f"Creating {num_envs} C++ parallel processes...")

    env = SubprocVecEnv([make_env() for _ in range(num_envs)])
    env = VecMonitor(env)
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
        learning_rate=0.0001,
        n_steps=256,
        batch_size=256,
        n_epochs=4,
        gamma=0.995,
        gae_lambda=0.95,
        clip_range=0.1,
        ent_coef=0.01,
        target_kl=0.03,
        policy_kwargs=policy_kwargs,
        tensorboard_log="./logs"
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=max(50000 // num_envs, 1),
        save_path="./models/checkpoints",
        name_prefix="ppo_space_invaders",
    )

    print("Starting training SmallGridCNN + PPO...")
    try:
        model.learn(total_timesteps=1000000, callback=checkpoint_callback)
    except KeyboardInterrupt:
        print("\nThe training was interrupted manually")
    finally:
        os.makedirs("./models", exist_ok=True)
        model.save("./models/ppo_space_invaders_final")
        env.close()


if __name__ == "__main__":
    train()
