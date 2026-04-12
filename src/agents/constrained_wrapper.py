"""Constrained PPO via reward shaping (Lagrangian relaxation).

Adds two penalty terms to the base reward:
  1. Smoothness: -λ_smooth × ||a_t - a_{t-1}||²
     Penalizes jerky control — large action changes between steps.
  2. Zeroness:  -λ_zero × ||a_t||²
     Pushes actions toward zero — prevents aggressive over-correction.

These implement the KL-based constraints from the Constrained PPO
literature as reward penalties, which is equivalent in expectation and
compatible with any RL library (no PPO source modification needed).

Usage:
    env = CursorEnv(noise_model=...)
    env = ConstrainedRewardWrapper(env, lambda_smooth=0.1, lambda_zero=0.05)
    model = PPO("MlpPolicy", env, ...)
"""
from __future__ import annotations

from typing import Optional

import gymnasium as gym
import numpy as np


class ConstrainedRewardWrapper(gym.Wrapper):
    """Adds smoothness and zeroness penalties to the reward.

    Args:
        env: base environment.
        lambda_smooth: weight for action smoothness penalty (default 0.1).
        lambda_zero: weight for action magnitude penalty (default 0.05).
    """

    def __init__(
        self,
        env: gym.Env,
        lambda_smooth: float = 0.1,
        lambda_zero: float = 0.05,
    ):
        super().__init__(env)
        self.lambda_smooth = lambda_smooth
        self.lambda_zero = lambda_zero
        self._prev_action: Optional[np.ndarray] = None

    def reset(self, **kwargs):
        self._prev_action = None
        obs, info = self.env.reset(**kwargs)
        return obs, info

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Store base reward for logging
        info["base_reward"] = reward

        # Smoothness penalty: ||a_t - a_{t-1}||²
        if self._prev_action is not None:
            smooth_penalty = float(np.sum((action - self._prev_action) ** 2))
        else:
            smooth_penalty = 0.0

        # Zeroness penalty: ||a_t||²
        zero_penalty = float(np.sum(action ** 2))

        penalty = (
            self.lambda_smooth * smooth_penalty
            + self.lambda_zero * zero_penalty
        )

        info["smooth_penalty"] = smooth_penalty
        info["zero_penalty"] = zero_penalty
        info["total_penalty"] = penalty

        reward -= penalty
        self._prev_action = action.copy()

        return obs, reward, terminated, truncated, info
