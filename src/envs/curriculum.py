"""Curriculum learning wrapper for CursorEnv.

Gradually increases task difficulty during RL training by:
  1. Shrinking the target radius (starts large, ends at true size)
  2. Moving targets further from center (starts close, ends at full distance)

Usage with stable-baselines3:
    env = CursorEnv(noise_model=...)
    env = CurriculumWrapper(env, total_steps=500_000)
    model = PPO("MlpPolicy", env, ...)
    model.learn(total_timesteps=500_000)
"""
from __future__ import annotations

from typing import Any, Optional

import gymnasium as gym
import numpy as np


class CurriculumWrapper(gym.Wrapper):
    """Wraps CursorEnv with progressive difficulty scheduling.

    Args:
        env: CursorEnv instance.
        total_steps: total training timesteps for scheduling.
        initial_radius_mult: multiplier on target_radius at start (default 4.0).
        initial_distance_mult: multiplier on target distance at start (default 0.3).
        warmup_frac: fraction of total_steps before curriculum kicks in (default 0.05).
        end_frac: fraction of total_steps by which difficulty reaches final (default 0.7).
    """

    def __init__(
        self,
        env: gym.Env,
        total_steps: int = 500_000,
        initial_radius_mult: float = 4.0,
        initial_distance_mult: float = 0.3,
        warmup_frac: float = 0.05,
        end_frac: float = 0.7,
    ):
        super().__init__(env)
        self.total_steps = total_steps
        self.initial_radius_mult = initial_radius_mult
        self.initial_distance_mult = initial_distance_mult
        self.warmup_frac = warmup_frac
        self.end_frac = end_frac

        self._base_target_radius = env.target_radius
        self._base_target_positions = [t.copy() for t in env.target_positions]
        self._global_step = 0

    @property
    def progress(self) -> float:
        """Returns curriculum progress in [0, 1]."""
        warmup_end = self.warmup_frac * self.total_steps
        curriculum_end = self.end_frac * self.total_steps
        if self._global_step < warmup_end:
            return 0.0
        if self._global_step >= curriculum_end:
            return 1.0
        return (self._global_step - warmup_end) / (curriculum_end - warmup_end)

    def _apply_curriculum(self):
        """Update env parameters based on current progress."""
        p = self.progress

        # Target radius: large → normal
        radius_mult = self.initial_radius_mult + (1.0 - self.initial_radius_mult) * p
        self.env.target_radius = self._base_target_radius * radius_mult

        # Target distance: close → full
        dist_mult = self.initial_distance_mult + (1.0 - self.initial_distance_mult) * p
        self.env.target_positions = [
            t * dist_mult for t in self._base_target_positions
        ]

    def reset(self, **kwargs) -> tuple[np.ndarray, dict]:
        self._apply_curriculum()
        obs, info = self.env.reset(**kwargs)
        info["curriculum_progress"] = self.progress
        info["curriculum_radius"] = self.env.target_radius
        return obs, info

    def step(self, action) -> tuple[np.ndarray, float, bool, bool, dict]:
        self._global_step += 1
        obs, reward, terminated, truncated, info = self.env.step(action)
        info["curriculum_progress"] = self.progress
        return obs, reward, terminated, truncated, info
