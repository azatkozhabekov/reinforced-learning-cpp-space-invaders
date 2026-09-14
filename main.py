import sys
from src.agent.train import train
from src.agent.evaluate import evaluate as test_best_model

if __name__ == "__main__":
    print("=== Space Invaders RL Agent ===")
    print("1. Начать обучение модели")
    print("2. Протестировать обученную модель")

    choice = input("Выберите вариант (1/2): ").strip()

    if choice == "1":
        train()
    elif choice == "2":
        test_best_model()
    else:
        print("Неверный ввод.")