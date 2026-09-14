import subprocess
import time
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import config


class SpaceInvadersEnv(gym.Env):
    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode

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

        # Заглушка для графика: если рендер выключен, глушим stdout бинарника C++
        stdout_dest = None if self.render_mode == "human" else subprocess.DEVNULL

        self.process = subprocess.Popen(
            config.WSL_COMMAND,
            stdin=subprocess.PIPE,
            stdout=stdout_dest,  # <-- DEVNULL уберёт весь вывод C++ из терминала!
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1
        )

        time.sleep(0.1)
        return self._get_observation(), {}

    def step(self, action):
        # 1. Проверяем, не умер ли C++ подпроцесс
        if self.process.poll() is not None:
            # Если процесс завершился, делаем перезапуск (reset)
            print("⚠️ C++ процесс неожиданно завершился. Перезапуск среды...")
            self.reset()
            return self.current_obs, -100.0, True, False, {"score": self.current_score}

        # 2. Безопасная запись в stdin
        try:
            self.process.stdin.write(f"{action}\n")
            self.process.stdin.flush()
        except (BrokenPipeError, OSError):
            print("⚠️ Ошибка записи в pipe (процесс C++ упал).")
            self.reset()
            return self.current_obs, -100.0, True, False, {"score": self.current_score}

        # 3. Чтение нового состояния из C++
        # (Убедитесь, что тут не зависает чтение, если C++ выдал EOF)
        self.current_obs = self._get_observation()
        new_score, terminated = self._read_game_status()

        reward = float(new_score - self.current_score)
        self.current_score = new_score

        if terminated:
            reward -= 100.0

        if self.render_mode == "human":
            self.render()

        return self.current_obs, reward, terminated, False, {"score": self.current_score}

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