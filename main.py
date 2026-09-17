import sys
from src.agent.train import train
from src.agent.evaluate import evaluate as test_best_model

if __name__ == "__main__":
    print("=== Space Invaders RL Agent ===")
    print("1. Train new model")
    print("2. Test the last trained model")

    choice = input("Enter 1 or 2: ").strip()

    if choice == "1":
        train()
    elif choice == "2":
        test_best_model()
    else:
        print("Invalid input")