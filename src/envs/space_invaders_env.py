import subprocess
import time
import gymnasium as gym
from gymnasium import spaces


class SpaceInvadersEnv(gym.Env):
    def __init__(self):
        super().__init__()

        # 1. Пространство действий (0: влево, 1: вправо, 2: выстрел, 3: ничего)
        self.action_space = spaces.Discrete(4)

        # 2. Пример пространства наблюдений (подстройте под свой формат вывода)
        self.observation_space = spaces.Box(low=0, high=255, shape=(20, 30), dtype=int)

        self.process = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Если старый процесс игры еще запущен — завершаем его
        if self.process is not None:
            self.process.terminate()
            self.process.wait()

        # 🚀 КОМАНДА ЗАПУСКА WSL И ИГРЫ
        # wsl.exe автоматически выполняет переход в папку и запуск бинарника
        wsl_cmd = [
            "wsl",
            "bash", "-c",
            "cd /mnt/c/C/PCC/MySpaceInvaders/build && ./SpaceInvaders --play"
        ]

        # Запускаем игру в фоновом процессе с перенаправлением ввода/вывода (stdin/stdout)
        self.process = subprocess.Popen(
            wsl_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )

        # Даем игре полсекунды на инициализацию
        time.sleep(0.5)

        # Считываем первое состояние игры
        observation = self._get_observation()
        info = {}
        return observation, info

    def step(self, action):
        # 1. Отправляем действие в WSL (в stdin C++ программы)
        # Например, отправляем цифру действия и перенос строки: "0\n"
        self.process.stdin.write(f"{action}\n")
        self.process.stdin.flush()

        # 2. Считываем новое состояние, награду и конец игры из stdout C++ программы
        observation = self._get_observation()
        reward = self._get_reward()
        terminated = self._check_if_terminal()
        truncated = False
        info = {}

        return observation, reward, terminated, truncated, info

    def _get_observation(self):
        # ЗДЕСЬ ВЫ СЧИТЫВАЕТЕ ДАННЫЕ ИЗ `self.process.stdout.readline()`
        # И преобразуете их в массив/матрицу для нейросети
        pass

    def _get_reward(self):
        # Логика получения награды из вывода программы
        return 0.0

    def _check_if_terminal(self):
        # Проверка, закончилась ли игра (проигрыш/победа)
        return False

    def close(self):
        if self.process is not None:
            self.process.terminate()