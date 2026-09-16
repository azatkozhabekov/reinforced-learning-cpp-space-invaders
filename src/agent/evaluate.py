from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

from src.envs.space_invaders_env import SpaceInvadersEnv


def make_env():
    def _init():
        return SpaceInvadersEnv(render_mode="human")

    return _init


def evaluate():
    model_path = "./models/ppo_space_invaders_final"

    print(f"Loading trained model from {model_path}.zip...")

    env = DummyVecEnv([make_env()])
    env = VecFrameStack(env, n_stack=4, channels_order="first")

    try:
        model = PPO.load(model_path, env=env)
    except ValueError as exc:
        env.close()
        print("\nCannot load this model with the current environment.")
        print("Most likely it was trained with the old 20x30 observation shape.")
        print("Run training again first, then start evaluation.")
        print(f"\nOriginal error: {exc}")
        return

    print("Starting evaluation. Press Ctrl+C to stop.\n")

    obs = env.reset()
    try:
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)

            if done[0]:
                score = info[0].get("score", 0)
                print(f"\nRound finished. Final score: {score}. Restarting...\n")
                obs = env.reset()
    except KeyboardInterrupt:
        print("\nEvaluation stopped.")
    finally:
        env.close()


if __name__ == "__main__":
    evaluate()
