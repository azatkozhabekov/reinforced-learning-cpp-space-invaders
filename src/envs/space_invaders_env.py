import subprocess
import time
import re
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import config

class SpaceInvadersEnv(gym.Env):
    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode
        self.current_step = 0
        self.max_steps = 500

        # 0: LEFT, 1: RIGHT, 2: SHOOT, 3: IDLE
        self.action_space = spaces.Discrete(4)

        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(config.GRID_HEIGHT, config.GRID_WIDTH),
            dtype=np.float32
        )

        self.process = None
        self.current_score = 0
        self.current_obs = np.zeros((config.GRID_HEIGHT, config.GRID_WIDTH), dtype=np.float32)

        self.mapping = {
            '#': 0.25, 'O': 0.50, 'X': 0.50, '>': 0.50,
            '+': 0.50, '|': 0.75, '^': 1.00, '<': 1.00,
        }

    def render(self):
        if self.render_mode == "human":
            time.sleep(0.04)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.current_score = 0

        if hasattr(self, 'process') and self.process is not None:
            try:
                if self.process.stdin: self.process.stdin.close()
                if self.process.stdout: self.process.stdout.close()
                self.process.terminate()
                self.process.wait(timeout=0.2)
            except Exception:
                pass

        # Для чтения матриц нам нужен PIPE даже в human-режиме, если хотим передавать obs модели
        stdout_dest = subprocess.PIPE if self.render_mode != "human" else None

        self.process = subprocess.Popen(
            config.WSL_COMMAND,
            stdin=subprocess.PIPE,
            stdout=stdout_dest,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1
        )

        time.sleep(0.05)

        if self.render_mode != "human":
            try:
                self.process.stdin.write("\n")
                self.process.stdin.flush()
            except Exception:
                pass
            self.current_obs = self._get_observation()
            return self.current_obs, {}

        return np.zeros((config.GRID_HEIGHT, config.GRID_WIDTH), dtype=np.float32), {}

    def step(self, action):
        self.current_step += 1

        # 1. Отправляем действие
        try:
            self.process.stdin.write(f"{action}\n")
            self.process.stdin.flush()
        except (BrokenPipeError, OSError):
            obs, _ = self.reset()
            return obs, -20.0, True, False, {"score": self.current_score}

        # 2. Получение кадра
        if self.render_mode != "human":
            self.current_obs, extracted_score, lives = self._get_observation_and_status()
            new_score = extracted_score if extracted_score is not None else self.current_score
            terminated = (lives is not None and lives <= 0)
        else:
            self.render()
            new_score = self.current_score
            terminated = False

        # 3. REWARD SHAPING (Синхронизировано с ACTIONS из config.py)
        reward = 0.0
        score_diff = new_score - self.current_score
        if score_diff > 0:
            reward += score_diff * 100.0  # Главный приоритет — сбивать врагов

        if action in [0, 1]:  # LEFT / RIGHT
            reward += 0.05    # Бонус за движение
        elif action == 2:     # SHOOT
            reward += 0.02    # Маленький бонус за выстрел
        elif action == 3:     # IDLE
            reward -= 0.05    # Штраф за бездействие

        if terminated:
            reward -= 20.0

        self.current_score = new_score
        truncated = self.current_step >= self.max_steps

        return self.current_obs, reward, terminated, truncated, {"score": self.current_score}

    def _get_observation_and_status(self):
        grid = np.zeros((config.GRID_HEIGHT, config.GRID_WIDTH), dtype=np.float32)
        if self.process is None or self.process.stdout is None:
            return grid, self.current_score, 3

        ansi_escape = re.compile(r'\x1b\[[0-9;]*m')
        lines = []

        # Читаем ровно один кадр до ключевого маркера
        while True:
            line = self.process.stdout.readline()
            if not line:
                break
            lines.append(line)
            if "Level:" in line:
                break

        valid_rows_read = 0
        extracted_score = None
        extracted_lives = None

        for line in lines:
            clean_line = ansi_escape.sub('', line).strip()

            if clean_line.startswith("Score:"):
                try:
                    parts = clean_line.split()
                    extracted_score = int(parts[1])
                except Exception:
                    pass
                continue

            if clean_line.startswith("Lives:"):
                try:
                    parts = clean_line.split()
                    extracted_lives = int(parts[1])
                except Exception:
                    pass
                continue

            if not clean_line or any(clean_line.startswith(p) for p in ["Level:", "Time:"]):
                continue

            if clean_line.startswith('-') or clean_line.startswith('|---'):
                continue

            if clean_line.startswith('|') and clean_line.endswith('|'):
                clean_line = clean_line[1:-1]

            if valid_rows_read < config.GRID_HEIGHT:
                for c in range(min(len(clean_line), config.GRID_WIDTH)):
                    char = clean_line[c]
                    if char in self.mapping:
                        grid[valid_rows_read, c] = self.mapping[char]
                valid_rows_read += 1

        return grid, extracted_score, extracted_lives

    def _get_observation(self):
        obs, _, _ = self._get_observation_and_status()
        return obs

    def close(self):
        if hasattr(self, 'process') and self.process is not None:
            try:
                if self.process.stdin:
                    self.process.stdin.close()
                if self.process.stdout:
                    self.process.stdout.close()
                self.process.terminate()
                self.process.wait(timeout=0.1)
            except Exception:
                pass