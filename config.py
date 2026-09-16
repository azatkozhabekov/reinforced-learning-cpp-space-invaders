import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
LOGS_DIR = os.path.join(BASE_DIR, "logs")

# Playable area inside the C++ border. main.cpp uses Game(40, 22), so the
# visible arena without the border is 38x20.
GRID_HEIGHT = 20
GRID_WIDTH = 38

# The C++ game reads raw keyboard-like chars from stdin.
ACTIONS = {
    0: "a",   # LEFT
    1: "d",   # RIGHT
    2: "w",   # SHOOT
    3: "",    # IDLE
}

WSL_COMMAND = [
    "wsl", "bash", "-c",
    "cd /mnt/c/C/PCC/MySpaceInvaders/build && ./SpaceInvaders --play"
]
