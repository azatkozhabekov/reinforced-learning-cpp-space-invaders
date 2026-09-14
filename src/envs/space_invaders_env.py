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
        self.current_step = 0
        self.max_steps = 2000

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

    def render(self):
        # Если render_mode == "human", C++ процесс сам пишет ASCII-кадр в stdout.
        # Метод render() нужен Gymnasium, чтобы не бросать NotImplementedError.
        if self.render_mode == "human":
            # Можно добавить задержку, чтобы игра в консоли не "летала" слишком быстро при просмотре
            time.sleep(0.05)
        else:
            pass

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0

        if hasattr(self, 'process') and self.process is not None:
            try:
                if self.process.stdin:
                    self.process.stdin.close()
                if self.process.stdout:
                    self.process.stdout.close()
                self.process.terminate()
                self.process.wait(timeout=0.2)
            except Exception:
                pass

        # Если human — выводим в консоль (None), если обучем — глушим в DEVNULL
        stdout_dest = None if self.render_mode == "human" else subprocess.DEVNULL

        self.process = subprocess.Popen(
            config.WSL_COMMAND,
            stdin=subprocess.PIPE,
            stdout=stdout_dest,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1
        )

        time.sleep(0.05)
        return self._get_observation(), {}

    def step(self, action):
        # 1. Отправляем действие в C++ процесс
        self.current_step += 1
        try:
            self.process.stdin.write(f"{action}\n")
            self.process.stdin.flush()
        except (BrokenPipeError, OSError):
            self.reset()
            return self.current_obs, -10.0, True, False, {"score": self.current_score}

        # 2. Получаем текущие данные из C++
        self.current_obs = self._get_observation()
        new_score, terminated = self._read_game_status()

        # -----------------------------------------------------------
        # 3. НАСТРОЙКА НАГРАД (REWARD SHAPING)
        # -----------------------------------------------------------
        reward = 0.0

        # А) Основная награда: за увеличение счета в игре (сбитый пришелец)
        score_diff = new_score - self.current_score
        if score_diff > 0:
            reward += score_diff * 50.0  # Умножаем на 10 для сильного стимула!

        # Б) Награда за выживание (Штраф за простой / Маленький бонус за шаг)
        # Помогает агенту не стоять на месте и быстрее двигаться
        reward -= 0.01

        # В) Штраф за выстрел впустую (если в action_space выстрел = действие 3)
        # Это отучит агента спамить стрельбой без остановки
        if action == 3:  # укажите номер действия стрельбы в вашей игре
            reward -= 100

        # Г) Жесткий штраф за поражение/смерть
        if terminated:
            reward -= 50.0

        # Обновляем текущий счет для следующего шага
        self.current_score = new_score
        # -----------------------------------------------------------

        # Рендер только если включен режим "human"
        if self.render_mode == "human":
            self.render()

        truncated = False
        if self.current_step >= self.max_steps:
            truncated = True

        return self.current_obs, reward, terminated, truncated, {"score": self.current_score}

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