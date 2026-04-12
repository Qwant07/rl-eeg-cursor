"""Behavior Cloning baseline — clone a proportional controller.

Collects (observation, action) pairs from a simple proportional controller
that always points toward the target, then trains a small MLP to replicate
that policy. Used as a non-RL baseline and optionally as warm-start for PPO.

Usage:
    bc = BehaviorCloning(env)
    bc.collect(n_episodes=500)
    bc.train(epochs=50)
    bc.save("results/S05/bc/policy.pt")

    # Evaluate
    success_rate = bc.evaluate(n_episodes=100)

    # Warm-start PPO (optional)
    ppo = PPO("MlpPolicy", env)
    bc.load_into_sb3(ppo)
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class BCPolicy(nn.Module):
    """Simple MLP for behavior cloning."""

    def __init__(self, obs_dim: int = 7, act_dim: int = 2, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, act_dim),
            nn.Tanh(),  # actions in [-1, 1]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BehaviorCloning:
    """Behavior Cloning from a proportional controller.

    Args:
        env: CursorEnv (or wrapped) instance.
        hidden: hidden layer size for the MLP (default 64).
        device: torch device (default "cpu").
    """

    def __init__(self, env, hidden: int = 64, device: str = "cpu"):
        self.env = env
        self.device = torch.device(device)
        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.shape[0]
        self.policy = BCPolicy(obs_dim, act_dim, hidden).to(self.device)
        self.obs_data: list = []
        self.act_data: list = []

    def _expert_action(self, obs: np.ndarray) -> np.ndarray:
        """Proportional controller: point toward target at max speed."""
        # obs layout: [cursor_x, cursor_y, target_x, target_y, vel_x, vel_y, time_rem]
        # These are normalized by workspace, so direction is just target - cursor
        cursor = obs[:2]
        target = obs[2:4]
        direction = target - cursor
        dist = np.linalg.norm(direction)
        if dist > 0.01:
            return (direction / dist).astype(np.float32)
        return np.zeros(2, dtype=np.float32)

    def collect(self, n_episodes: int = 500, seed_offset: int = 0):
        """Collect expert demonstrations from proportional controller."""
        self.obs_data.clear()
        self.act_data.clear()
        for ep in range(n_episodes):
            obs, _ = self.env.reset(seed=ep + seed_offset)
            for _ in range(self.env.unwrapped.max_steps):
                action = self._expert_action(obs)
                self.obs_data.append(obs.copy())
                self.act_data.append(action.copy())
                obs, _, terminated, truncated, _ = self.env.step(action)
                if terminated or truncated:
                    break

    def train(
        self,
        epochs: int = 50,
        batch_size: int = 256,
        lr: float = 1e-3,
    ) -> list[float]:
        """Train the BC policy on collected data.

        Returns:
            List of per-epoch MSE losses.
        """
        obs_t = torch.tensor(np.array(self.obs_data), dtype=torch.float32)
        act_t = torch.tensor(np.array(self.act_data), dtype=torch.float32)
        dataset = TensorDataset(obs_t, act_t)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        criterion = nn.MSELoss()

        losses = []
        for epoch in range(epochs):
            total_loss = 0.0
            for obs_b, act_b in loader:
                obs_b = obs_b.to(self.device)
                act_b = act_b.to(self.device)
                pred = self.policy(obs_b)
                loss = criterion(pred, act_b)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(obs_b)
            epoch_loss = total_loss / len(dataset)
            losses.append(epoch_loss)
        return losses

    def predict(self, obs: np.ndarray) -> np.ndarray:
        """Predict action from observation."""
        self.policy.eval()
        with torch.no_grad():
            obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
            action = self.policy(obs_t).cpu().numpy()[0]
        return action

    def evaluate(self, n_episodes: int = 100, seed_offset: int = 5000) -> dict:
        """Evaluate the BC policy.

        Returns:
            Dict with success_rate, avg_steps, etc.
        """
        self.policy.eval()
        successes, steps_list = 0, []
        for ep in range(n_episodes):
            obs, _ = self.env.reset(seed=ep + seed_offset)
            for step in range(self.env.unwrapped.max_steps):
                action = self.predict(obs)
                obs, _, terminated, truncated, _ = self.env.step(action)
                if terminated:
                    successes += 1
                    steps_list.append(step + 1)
                    break
                if truncated:
                    steps_list.append(step + 1)
                    break
        return {
            "success_rate": successes / n_episodes,
            "successes": successes,
            "n_episodes": n_episodes,
            "avg_steps": float(np.mean(steps_list)),
        }

    def save(self, path: str):
        """Save BC policy weights."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.policy.state_dict(), path)

    def load(self, path: str):
        """Load BC policy weights."""
        self.policy.load_state_dict(
            torch.load(path, map_location=self.device, weights_only=True)
        )
