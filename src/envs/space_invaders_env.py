import subprocess
import time
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import config


class SpaceInvadersEnv(gym.Env):
    def __init__(self):
        super().__init__()

        # 4 действия: Влево, Вправо, Выстрел, Простой
        self.action_space = spaces.Discrete(4)

        # Игровое поле в виде матрицы символов/чисел
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(config.GRID_HEIGHT, config.GRID_WIDTH),
            dtype=np.uint8
        )

        self.process = None
        self.current_score = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if self.process is not None:
            self.process.terminate()
            self.process.wait()

        self.process = subprocess.Popen(
            config.WSL_COMMAND,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )

        time.sleep(0.2)
        self.current_score = 0

        obs = self._get_observation()
        return obs, {}

    def step(self, action):
        if self.process.poll() is not None:
            return np.zeros((config.GRID_HEIGHT, config.GRID_WIDTH), dtype=np.uint8), 0.0, True, False, {}

        # 1. Отправляем действие в C++ через stdin
        self.process.stdin.write(f"{action}\n")
        self.process.stdin.flush()

        # 2. Получаем обновленные данные из stdout
        obs = self._get_observation()
        new_score, terminated = self._read_game_status()

        # 3. Награда за получение очков (Максимизация счета)
        score_diff = new_score - self.current_score
        self.current_score = new_score

        # За выбитые очки даём положительный Reward, за смерть — штраф
        reward = float(score_diff)
        if terminated:
            reward -= 100.0  # Штраф за законченную попытку

        return obs, reward, terminated, False, {"score": self.current_score}

    def _get_observation(self):
        # Ожидается, что C++ программа выводит матрицу (например, 20 строк)
        # Если вы пока не настроили вывод кадра — возвращаем пустую матрицу
        matrix = np.zeros((config.GRID_HEIGHT, config.GRID_WIDTH), dtype=np.uint8)
        try:
            # Чтение кадра из stdout процесса C++
            # Пример парсинга строк:
            # for i in range(config.GRID_HEIGHT):
            #     line = self.process.stdout.readline().strip()
            #     matrix[i] = [int(x) for x in line.split()]
            pass
        except Exception:
            pass
        return matrix

    def _read_game_status(self):
        # Ожидается чтение счета и флага окончания игры из C++ программы
        # Формат вывода в C++: "SCORE: 150 GAME_OVER: 0"
        score = self.current_score
        game_over = False
        try:
            # line = self.process.stdout.readline().strip()
            # score, game_over = parse(line)
            pass
        except Exception:
            pass
        return score, game_over

    def close(self):
        if self.process is not None:
            self.process.terminate()