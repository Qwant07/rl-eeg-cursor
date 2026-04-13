"""Formal BCI evaluation metrics for cursor control.

Implements standard metrics from the BCI literature:
  - Fitts' Throughput (TP): bits/s, measures speed-accuracy tradeoff
  - Path Efficiency: ratio of ideal to actual path length
  - Action Smoothness: mean squared jerk of control signal
  - Shannon Information Transfer Rate (ITR): bits/trial
  - Movement time and success rate statistics

References:
  - Fitts (1954), Soukoreff & MacKenzie (2004) for throughput
  - Wolpaw et al. (2002), Yuan et al. (2013) for BCI-specific ITR
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class EpisodeMetrics:
    """Metrics computed from a single episode trajectory."""
    success: bool
    movement_time: float          # seconds
    steps: int
    fitts_id: float               # bits (effective index of difficulty)
    fitts_tp: float               # bits/s (throughput)
    path_efficiency: float        # [0, 1]
    action_smoothness: float      # mean ||a_t - a_{t-1}||²
    mean_action_magnitude: float  # mean ||a_t||
    final_distance: float         # distance to target at episode end
    target_distance: float        # initial distance cursor→target
    cursor_trajectory: np.ndarray = field(repr=False)  # (T, 2)
    actions: np.ndarray = field(repr=False)             # (T, 2)


def fitts_throughput(
    target_distance: float,
    target_width: float,
    movement_time: float,
) -> tuple[float, float]:
    """Compute Fitts' Index of Difficulty and Throughput.

    Uses the Shannon formulation (ISO 9241-9):
        ID = log2(D / W + 1)
        TP = ID / MT

    Args:
        target_distance: distance from start to target center (D).
        target_width: effective target width, typically 2 * radius (W).
        movement_time: time to acquire target in seconds (MT).

    Returns:
        (index_of_difficulty, throughput) — (bits, bits/s).
    """
    if target_width <= 0:
        raise ValueError("target_width must be positive")
    id_bits = np.log2(target_distance / target_width + 1.0)
    if movement_time <= 0:
        return id_bits, 0.0
    tp = id_bits / movement_time
    return float(id_bits), float(tp)


def path_efficiency(
    cursor_positions: np.ndarray,
    start_pos: np.ndarray,
    target_pos: np.ndarray,
) -> float:
    """Ratio of straight-line distance to actual path length.

    A perfectly straight path gives efficiency = 1.0.
    Longer, more tortuous paths give efficiency < 1.0.

    Args:
        cursor_positions: (T, 2) array of cursor positions over time.
        start_pos: (2,) initial cursor position.
        target_pos: (2,) target position.

    Returns:
        Path efficiency in (0, 1]. Returns 1.0 if no movement.
    """
    straight_line = float(np.linalg.norm(target_pos - start_pos))
    if straight_line < 1e-8:
        return 1.0

    # Actual path length = sum of step-to-step distances
    diffs = np.diff(cursor_positions, axis=0)
    actual_length = float(np.sum(np.linalg.norm(diffs, axis=1)))

    if actual_length < 1e-8:
        return 1.0

    return min(straight_line / actual_length, 1.0)


def action_smoothness(actions: np.ndarray) -> float:
    """Mean squared change in actions (jerkiness).

    Lower values indicate smoother control.
        smoothness = mean_t ||a_t - a_{t-1}||²

    Args:
        actions: (T, 2) array of actions over the episode.

    Returns:
        Mean squared jerk. 0.0 if fewer than 2 actions.
    """
    if len(actions) < 2:
        return 0.0
    diffs = np.diff(actions, axis=0)
    return float(np.mean(np.sum(diffs ** 2, axis=1)))


def shannon_itr(
    n_targets: int,
    accuracy: float,
    trial_time: float,
) -> float:
    """Shannon Information Transfer Rate (bits/min).

    From Wolpaw et al. (2002):
        B = log2(N) + P*log2(P) + (1-P)*log2((1-P)/(N-1))
        ITR = B / trial_time * 60

    Args:
        n_targets: number of possible targets (N).
        accuracy: fraction of correct selections (P), in [0, 1].
        trial_time: average time per trial in seconds.

    Returns:
        Information transfer rate in bits/min.
    """
    if n_targets < 2:
        raise ValueError("Need at least 2 targets for ITR")
    if trial_time <= 0:
        return 0.0

    # Clamp accuracy to avoid log(0)
    p = np.clip(accuracy, 1e-8, 1.0 - 1e-8)
    n = n_targets

    bits_per_trial = (
        np.log2(n)
        + p * np.log2(p)
        + (1 - p) * np.log2((1 - p) / (n - 1))
    )
    # Clamp to non-negative (can go negative if accuracy < 1/N)
    bits_per_trial = max(bits_per_trial, 0.0)

    return float(bits_per_trial / trial_time * 60.0)


def compute_episode_metrics(
    cursor_positions: np.ndarray,
    actions: np.ndarray,
    start_pos: np.ndarray,
    target_pos: np.ndarray,
    target_radius: float,
    dt: float,
    success: bool,
) -> EpisodeMetrics:
    """Compute all metrics for a single episode.

    Args:
        cursor_positions: (T, 2) cursor trajectory.
        actions: (T, 2) actions taken.
        start_pos: (2,) initial cursor position.
        target_pos: (2,) target position.
        target_radius: target acquisition radius.
        dt: time step in seconds.
        success: whether the target was acquired.

    Returns:
        EpisodeMetrics with all computed values.
    """
    steps = len(actions)
    mt = steps * dt
    target_dist = float(np.linalg.norm(target_pos - start_pos))
    target_width = 2.0 * target_radius

    fid, ftp = fitts_throughput(target_dist, target_width, mt)
    pe = path_efficiency(cursor_positions, start_pos, target_pos)
    smooth = action_smoothness(actions)
    mean_mag = float(np.mean(np.linalg.norm(actions, axis=1))) if len(actions) > 0 else 0.0
    final_dist = float(np.linalg.norm(cursor_positions[-1] - target_pos)) if len(cursor_positions) > 0 else 0.0

    return EpisodeMetrics(
        success=success,
        movement_time=mt,
        steps=steps,
        fitts_id=fid,
        fitts_tp=ftp,
        path_efficiency=pe,
        action_smoothness=smooth,
        mean_action_magnitude=mean_mag,
        final_distance=final_dist,
        target_distance=target_dist,
        cursor_trajectory=cursor_positions,
        actions=actions,
    )


def aggregate_metrics(
    episodes: list[EpisodeMetrics],
) -> dict:
    """Aggregate metrics across episodes with mean and 95% CI.

    Args:
        episodes: list of EpisodeMetrics from multiple episodes.

    Returns:
        Dict with keys like "success_rate", "movement_time_mean",
        "movement_time_ci", etc.
    """
    n = len(episodes)
    if n == 0:
        return {}

    success_arr = np.array([e.success for e in episodes], dtype=float)
    success_rate = float(np.mean(success_arr))

    # Only compute time-based metrics on successful episodes
    success_eps = [e for e in episodes if e.success]
    n_success = len(success_eps)

    result = {
        "n_episodes": n,
        "n_success": n_success,
        "success_rate": success_rate,
    }

    if n_success > 0:
        for key, getter in [
            ("movement_time", lambda e: e.movement_time),
            ("steps", lambda e: e.steps),
            ("fitts_id", lambda e: e.fitts_id),
            ("fitts_tp", lambda e: e.fitts_tp),
            ("path_efficiency", lambda e: e.path_efficiency),
            ("action_smoothness", lambda e: e.action_smoothness),
            ("mean_action_magnitude", lambda e: e.mean_action_magnitude),
        ]:
            values = np.array([getter(e) for e in success_eps])
            mean = float(np.mean(values))
            std = float(np.std(values, ddof=1)) if n_success > 1 else 0.0
            ci = 1.96 * std / np.sqrt(n_success) if n_success > 1 else 0.0
            result[f"{key}_mean"] = mean
            result[f"{key}_std"] = std
            result[f"{key}_ci95"] = float(ci)

    # All episodes (including failures) for final distance
    final_dists = np.array([e.final_distance for e in episodes])
    result["final_distance_mean"] = float(np.mean(final_dists))

    return result
