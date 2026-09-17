import queue
import re
import subprocess
import sys
import threading
import time

import gymnasium as gym
import numpy as np
from gymnasium import spaces

import config


class SpaceInvadersEnv(gym.Env):
    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode
        self.current_step = 0
        self.max_steps = 2000

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(1, config.GRID_HEIGHT, config.GRID_WIDTH),
            dtype=np.float32,
        )

        self.process = None
        self.stdout_queue = None
        self.reader_thread = None
        self.current_score = 0
        self.current_level = 1
        self.frame_level = 1
        self.steps_since_score = 0
        self.frame_features = self._empty_features()
        self.prev_alignment_dist = None
        self.prev_danger = 0
        self.prev_player_bullet_count = 0
        self.current_obs = np.zeros(
            (1, config.GRID_HEIGHT, config.GRID_WIDTH), dtype=np.float32
        )

        self.mapping = {
            "#": 0.20,
            "O": 0.45,
            "X": 0.45,
            ">": 0.45,
            "+": 0.45,
            "*": 0.45,
            "_": 0.45,
            "x": 0.70,
            "|": 0.80,
            "^": 1.00,
            "<": 1.00,
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.current_score = 0
        self.current_level = 1
        self.frame_level = 1
        self.steps_since_score = 0
        self.frame_features = self._empty_features()
        self.prev_alignment_dist = None
        self.prev_danger = 0
        self.prev_player_bullet_count = 0
        self._stop_process()

        self.stdout_queue = queue.Queue()
        self.process = subprocess.Popen(
            config.WSL_COMMAND,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        self.reader_thread = threading.Thread(target=self._read_stdout, daemon=True)
        self.reader_thread.start()

        time.sleep(0.05)
        self.current_obs, extracted_score, _ = self._get_observation_and_status(timeout=2.0)
        if extracted_score is not None:
            self.current_score = extracted_score
        self.current_level = self.frame_level
        self.prev_alignment_dist = self._alignment_distance(self.frame_features)
        self.prev_danger = self._danger_level(self.frame_features)
        self.prev_player_bullet_count = len(self.frame_features["player_bullets"])
        return self.current_obs, {}

    def step(self, action):
        self.current_step += 1

        try:
            action_index = int(action)
            command = config.ACTIONS.get(action_index, "")
            self.process.stdin.write(f"{command}\n")
            self.process.stdin.flush()
        except (BrokenPipeError, OSError, ValueError, TypeError):
            return self.current_obs, -20.0, True, False, {"score": self.current_score}

        self.current_obs, extracted_score, lives = self._get_observation_and_status()
        new_score = extracted_score if extracted_score is not None else self.current_score
        score_diff = new_score - self.current_score
        level_diff = max(0, self.frame_level - self.current_level)

        reward = self._shape_reward(action_index, score_diff)
        if score_diff > 0:
            self.steps_since_score = 0
            reward += float(score_diff)
        else:
            self.steps_since_score += 1

        terminated = lives is not None and lives <= 0
        if terminated:
            reward -= 30.0

        if level_diff > 0:
            reward += 200.0 * level_diff

        self.current_score = new_score
        self.current_level = self.frame_level
        self.prev_alignment_dist = self._alignment_distance(self.frame_features)
        self.prev_danger = self._danger_level(self.frame_features)
        self.prev_player_bullet_count = len(self.frame_features["player_bullets"])
        truncated = self.current_step >= self.max_steps
        return self.current_obs, reward, terminated, truncated, {
            "score": self.current_score,
            "level": self.current_level,
        }

    def _read_stdout(self):
        try:
            for line in self.process.stdout:
                self.stdout_queue.put(line)
        except Exception:
            pass

    def _get_observation_and_status(self, timeout=1.0):
        grid = np.zeros((config.GRID_HEIGHT, config.GRID_WIDTH), dtype=np.float32)
        if self.process is None or self.stdout_queue is None:
            return np.expand_dims(grid, axis=0), self.current_score, 0

        lines = []
        got_complete_frame = False
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                line = self.stdout_queue.get(timeout=0.05)
            except queue.Empty:
                if self.process.poll() is not None:
                    break
                continue

            lines.append(line)
            if self.render_mode == "human":
                print(line, end="")
                sys.stdout.flush()

            if "Level:" in line:
                got_complete_frame = True
                break

        if not lines:
            return self.current_obs, self.current_score, 0

        if not got_complete_frame:
            return self.current_obs, self.current_score, 0

        return self._parse_frame(lines)

    def _parse_frame(self, lines):
        grid = np.zeros((config.GRID_HEIGHT, config.GRID_WIDTH), dtype=np.float32)
        ansi_escape = re.compile(r"\x1b\[[0-9;]*m")
        valid_rows_read = 0
        extracted_score = None
        extracted_lives = None
        extracted_level = self.current_level
        features = self._empty_features()

        for line in lines:
            clean_line = ansi_escape.sub("", line).lstrip("\r").rstrip("\n")
            stripped_line = clean_line.strip()

            if stripped_line.startswith("Score:"):
                try:
                    extracted_score = int(stripped_line.split()[1])
                except (IndexError, ValueError):
                    pass
                continue

            if stripped_line.startswith("Lives:"):
                try:
                    extracted_lives = int(stripped_line.split()[1])
                except (IndexError, ValueError):
                    pass
                continue

            if stripped_line.startswith("Level:"):
                try:
                    extracted_level = int(stripped_line.split()[1])
                except (IndexError, ValueError):
                    pass
                continue

            if not stripped_line or stripped_line.startswith(("Level:", "Time:")):
                continue

            if set(stripped_line) == {"-"}:
                continue

            if clean_line.startswith("|") and clean_line.endswith("|"):
                clean_line = clean_line[1:-1]

            if valid_rows_read < config.GRID_HEIGHT:
                for c in range(min(len(clean_line), config.GRID_WIDTH)):
                    value = self.mapping.get(clean_line[c])
                    if value is not None:
                        grid[valid_rows_read, c] = value
                    self._collect_feature(features, clean_line[c], valid_rows_read, c)
                valid_rows_read += 1

        self.frame_features = features
        self.frame_level = extracted_level
        return np.expand_dims(grid, axis=0), extracted_score, extracted_lives

    def _empty_features(self):
        return {
            "player": None,
            "enemies": [],
            "enemy_bullets": [],
            "player_bullets": [],
        }

    def _collect_feature(self, features, char, row, col):
        if char == "^":
            features["player"] = (row, col)
        elif char == "x":
            features["enemy_bullets"].append((row, col))
        elif char == "|":
            features["player_bullets"].append((row, col))
        elif char in {"X", "O", "+", "*", ">", "<", "_"} and row < 14:
            features["enemies"].append((row, col))

    def _alignment_distance(self, features):
        player = features.get("player")
        enemies = features.get("enemies", [])
        if player is None or not enemies:
            return None

        _, player_x = player
        return min(abs(player_x - enemy_x) for _, enemy_x in enemies)

    def _danger_level(self, features):
        player = features.get("player")
        if player is None:
            return 0

        player_y, player_x = player
        danger = 0
        for bullet_y, bullet_x in features.get("enemy_bullets", []):
            if bullet_y <= player_y and abs(bullet_x - player_x) <= 1:
                danger += 1
            elif bullet_y <= player_y and abs(bullet_x - player_x) <= 3:
                danger += 0.35
        return danger

    def _shape_reward(self, action, score_diff):
        reward = -0.002
        alignment_dist = self._alignment_distance(self.frame_features)
        danger = self._danger_level(self.frame_features)

        if action == 3:
            reward -= 0.01

        if action in (0, 1) and self.prev_alignment_dist is not None and alignment_dist is not None:
            if alignment_dist < self.prev_alignment_dist:
                reward += 0.015
            elif alignment_dist > self.prev_alignment_dist:
                reward -= 0.01

        if action == 2:
            bullet_count = len(self.frame_features["player_bullets"])
            shot_is_active = bullet_count > self.prev_player_bullet_count
            if shot_is_active and alignment_dist is not None and alignment_dist <= 2:
                reward += 0.025
            elif shot_is_active:
                reward += 0.005
            else:
                reward -= 0.01

        if danger < self.prev_danger:
            reward += 0.02
        elif danger > self.prev_danger:
            reward -= 0.03

        if self.steps_since_score > 0 and self.steps_since_score % 200 == 0:
            reward -= 0.2

        if score_diff < 0:
            reward -= 1.0

        return reward

    def _get_observation(self):
        obs, _, _ = self._get_observation_and_status()
        return obs

    def _stop_process(self):
        if self.process is None:
            return

        try:
            if self.process.stdin:
                self.process.stdin.close()
            if self.process.stdout:
                self.process.stdout.close()
            self.process.terminate()
            self.process.wait(timeout=0.2)
        except Exception:
            try:
                self.process.kill()
            except Exception:
                pass
        finally:
            self.process = None

    def close(self):
        self._stop_process()
