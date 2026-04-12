"""Gymnasium environment for 2D cursor reaching with decoder noise.

The RL agent acts as the brain controller: it outputs intended cursor
velocity, which passes through a noisy decoder channel (modelled by
DecoderNoiseModel). The cursor moves based on the decoded output.

The agent learns to compensate for the decoder's gain attenuation,
bias, and noise to reach targets accurately.

Observation (7D):
    [cursor_x, cursor_y, target_x, target_y,
     last_decoded_vx, last_decoded_vy, time_remaining]
    All values normalized to roughly [-1, 1].

Action (2D continuous):
    [intended_vx, intended_vy] in [-1, 1], scaled by vel_scale.

Reward:
    -distance_to_target each step, +success_bonus on target acquisition.
"""
from __future__ import annotations

from collections import deque
from typing import Any, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from src.envs.noise_model import DecoderNoiseModel


class CursorEnv(gym.Env):
    """2D cursor reaching task with decoder noise channel.

    Args:
        noise_model: DecoderNoiseModel instance, or None for perfect control.
        workspace: half-width of the square workspace (default 0.5).
        target_radius: radius for target acquisition (default 0.05).
        max_steps: maximum steps per episode (default 200).
        dt: time step in seconds (default 0.1).
        vel_scale: maps action [-1,1] to velocity (default 0.5).
        dwell_steps: steps cursor must stay in target to succeed (default 4).
        latency_steps: sensorimotor delay in steps (default 0, ~0 is 200ms at
            dt=0.1 would be 2 steps).
        success_bonus: reward added on target acquisition (default 10.0).
        target_positions: list of (x,y) positions for center-out targets.
            If None, uses 8 uniformly spaced targets at 80% workspace radius.
        randomize_start: if True, cursor starts at random position (default False).
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        noise_model: Optional[DecoderNoiseModel] = None,
        workspace: float = 0.5,
        target_radius: float = 0.05,
        max_steps: int = 200,
        dt: float = 0.1,
        vel_scale: float = 0.5,
        dwell_steps: int = 4,
        latency_steps: int = 0,
        success_bonus: float = 10.0,
        target_positions: Optional[list] = None,
        randomize_start: bool = False,
    ):
        super().__init__()
        self.noise_model = noise_model
        self.workspace = workspace
        self.target_radius = target_radius
        self.max_steps = max_steps
        self.dt = dt
        self.vel_scale = vel_scale
        self.dwell_steps = dwell_steps
        self.latency_steps = latency_steps
        self.success_bonus = success_bonus
        self.randomize_start = randomize_start

        if target_positions is not None:
            self.target_positions = [np.array(t, dtype=np.float32) for t in target_positions]
        else:
            # 8 center-out targets at 80% of workspace radius
            r = 0.8 * workspace
            self.target_positions = [
                np.array([r * np.cos(a), r * np.sin(a)], dtype=np.float32)
                for a in np.linspace(0, 2 * np.pi, 8, endpoint=False)
            ]

        # Observation: cursor(2) + target(2) + last_decoded_vel(2) + time_remaining(1)
        self.observation_space = spaces.Box(
            low=-2.0, high=2.0, shape=(7,), dtype=np.float32
        )
        # Action: intended velocity (2D)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )

        # Internal state (set in reset)
        self.cursor_pos = np.zeros(2, dtype=np.float32)
        self.target_pos = np.zeros(2, dtype=np.float32)
        self.last_decoded_vel = np.zeros(2, dtype=np.float32)
        self.step_count = 0
        self.dwell_count = 0
        self.vel_buffer: deque = deque(maxlen=max(1, latency_steps + 1))
        self._rng = np.random.default_rng()

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
            if self.noise_model is not None:
                self.noise_model.rng = np.random.default_rng(seed + 1)

        # Cursor start position
        if self.randomize_start:
            self.cursor_pos = self._rng.uniform(
                -self.workspace, self.workspace, size=2
            ).astype(np.float32)
        else:
            self.cursor_pos = np.zeros(2, dtype=np.float32)

        # Random target from the set
        idx = self._rng.integers(len(self.target_positions))
        self.target_pos = self.target_positions[idx].copy()

        self.last_decoded_vel = np.zeros(2, dtype=np.float32)
        self.step_count = 0
        self.dwell_count = 0
        self.vel_buffer = deque(
            [np.zeros(2, dtype=np.float32)] * (self.latency_steps + 1),
            maxlen=self.latency_steps + 1,
        )

        return self._get_obs(), self._get_info()

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        action = np.asarray(action, dtype=np.float32).clip(-1.0, 1.0)

        # Scale action to velocity
        intended_vel = action * self.vel_scale

        # Pass through decoder noise model
        if self.noise_model is not None:
            decoded_vel = self.noise_model(intended_vel).astype(np.float32)
        else:
            decoded_vel = intended_vel.copy()

        # Apply latency
        if self.latency_steps > 0:
            self.vel_buffer.append(decoded_vel)
            effective_vel = self.vel_buffer[0]
        else:
            effective_vel = decoded_vel

        self.last_decoded_vel = effective_vel

        # Update cursor position
        self.cursor_pos = self.cursor_pos + effective_vel * self.dt
        self.cursor_pos = self.cursor_pos.clip(
            -self.workspace, self.workspace
        )

        # Distance to target
        dist = float(np.linalg.norm(self.cursor_pos - self.target_pos))

        # Dwell check
        in_target = dist < self.target_radius
        if in_target:
            self.dwell_count += 1
        else:
            self.dwell_count = 0

        # Reward: negative distance + bonus on acquisition
        reward = -dist
        terminated = False
        if self.dwell_count >= self.dwell_steps:
            reward += self.success_bonus
            terminated = True

        self.step_count += 1
        truncated = self.step_count >= self.max_steps

        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def _get_obs(self) -> np.ndarray:
        """Build observation vector, normalized to roughly [-1, 1]."""
        time_remaining = 1.0 - self.step_count / self.max_steps
        obs = np.array([
            self.cursor_pos[0] / self.workspace,
            self.cursor_pos[1] / self.workspace,
            self.target_pos[0] / self.workspace,
            self.target_pos[1] / self.workspace,
            self.last_decoded_vel[0] / self.vel_scale,
            self.last_decoded_vel[1] / self.vel_scale,
            time_remaining,
        ], dtype=np.float32)
        return obs

    def _get_info(self) -> dict:
        dist = float(np.linalg.norm(self.cursor_pos - self.target_pos))
        return {
            "distance": dist,
            "in_target": dist < self.target_radius,
            "dwell_count": self.dwell_count,
            "step": self.step_count,
            "cursor_pos": self.cursor_pos.copy(),
            "target_pos": self.target_pos.copy(),
        }
