import os

# Пути к директориям
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
LOGS_DIR = os.path.join(BASE_DIR, "logs")

# Настройки игры
GRID_HEIGHT = 20
GRID_WIDTH = 30
ACTIONS = {
    0: "LEFT",
    1: "RIGHT",
    2: "SHOOT",
    3: "IDLE",
}

# Команда запуска C++ игры в WSL
WSL_COMMAND = [
    "wsl", "bash", "-c",
    "cd /mnt/c/C/PCC/MySpaceInvaders/build && ./SpaceInvaders --play"
]