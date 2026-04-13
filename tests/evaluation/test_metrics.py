"""Tests for BCI evaluation metrics."""
import numpy as np
import pytest

from src.evaluation.metrics import (
    fitts_throughput,
    path_efficiency,
    action_smoothness,
    shannon_itr,
    compute_episode_metrics,
    aggregate_metrics,
    EpisodeMetrics,
)


# ── Fitts' Throughput ────────────────────────────────────────────────

class TestFittsTP:
    def test_known_values(self):
        # D=0.4, W=0.1 → ID = log2(0.4/0.1 + 1) = log2(5) ≈ 2.322
        # MT = 2.0s → TP = 2.322 / 2.0 ≈ 1.161
        fid, ftp = fitts_throughput(0.4, 0.1, 2.0)
        assert fid == pytest.approx(np.log2(5), abs=1e-6)
        assert ftp == pytest.approx(np.log2(5) / 2.0, abs=1e-6)

    def test_zero_distance(self):
        fid, ftp = fitts_throughput(0.0, 0.1, 1.0)
        assert fid == pytest.approx(0.0)  # log2(0/0.1 + 1) = log2(1) = 0
        assert ftp == pytest.approx(0.0)

    def test_zero_movement_time(self):
        fid, ftp = fitts_throughput(0.4, 0.1, 0.0)
        assert fid > 0
        assert ftp == 0.0

    def test_invalid_width(self):
        with pytest.raises(ValueError):
            fitts_throughput(0.4, 0.0, 1.0)

    def test_higher_difficulty_with_smaller_target(self):
        fid_large, _ = fitts_throughput(0.4, 0.2, 1.0)
        fid_small, _ = fitts_throughput(0.4, 0.05, 1.0)
        assert fid_small > fid_large


# ── Path Efficiency ──────────────────────────────────────────────────

class TestPathEfficiency:
    def test_straight_line(self):
        positions = np.array([[0, 0], [0.1, 0], [0.2, 0], [0.3, 0], [0.4, 0]])
        pe = path_efficiency(positions, np.array([0, 0]), np.array([0.4, 0]))
        assert pe == pytest.approx(1.0, abs=1e-6)

    def test_detour_reduces_efficiency(self):
        # Go right, up, then to target — longer path
        positions = np.array([
            [0.0, 0.0], [0.2, 0.0], [0.2, 0.2], [0.4, 0.2], [0.4, 0.0],
        ])
        pe = path_efficiency(positions, np.array([0, 0]), np.array([0.4, 0]))
        assert pe < 1.0
        assert pe > 0.0

    def test_no_movement(self):
        positions = np.array([[0.0, 0.0], [0.0, 0.0]])
        pe = path_efficiency(positions, np.array([0, 0]), np.array([0.4, 0]))
        assert pe == 1.0  # edge case: no movement

    def test_start_equals_target(self):
        positions = np.array([[0.0, 0.0], [0.1, 0.0]])
        pe = path_efficiency(positions, np.array([0, 0]), np.array([0, 0]))
        assert pe == 1.0


# ── Action Smoothness ────────────────────────────────────────────────

class TestActionSmoothness:
    def test_constant_action(self):
        actions = np.array([[0.5, 0.5]] * 10)
        assert action_smoothness(actions) == pytest.approx(0.0)

    def test_alternating_action(self):
        actions = np.array([[1, 0], [-1, 0], [1, 0], [-1, 0]])
        # Diffs: [-2,0], [2,0], [-2,0] → squared norms: 4, 4, 4 → mean = 4
        assert action_smoothness(actions) == pytest.approx(4.0)

    def test_single_action(self):
        actions = np.array([[0.5, 0.5]])
        assert action_smoothness(actions) == 0.0

    def test_empty_actions(self):
        actions = np.zeros((0, 2))
        assert action_smoothness(actions) == 0.0

    def test_gradual_ramp(self):
        # Smooth ramp: [0,0] → [0.1,0] → [0.2,0] → ... → [1.0,0]
        actions = np.column_stack([np.linspace(0, 1, 11), np.zeros(11)])
        smooth = action_smoothness(actions)
        # Each diff is [0.1, 0] → squared norm = 0.01 → mean = 0.01
        assert smooth == pytest.approx(0.01, abs=1e-6)


# ── Shannon ITR ──────────────────────────────────────────────────────

class TestShannonITR:
    def test_perfect_accuracy(self):
        # 8 targets, 100% accuracy, 2s/trial
        # B = log2(8) + 1*log2(1) + 0 = 3 bits
        # ITR = 3 / 2 * 60 = 90 bits/min
        itr = shannon_itr(8, 1.0 - 1e-8, 2.0)
        assert itr == pytest.approx(90.0, abs=0.1)

    def test_chance_accuracy(self):
        # At chance level (1/N), ITR ≈ 0
        itr = shannon_itr(8, 1.0 / 8, 2.0)
        assert itr == pytest.approx(0.0, abs=0.5)

    def test_higher_accuracy_higher_itr(self):
        itr_low = shannon_itr(8, 0.5, 2.0)
        itr_high = shannon_itr(8, 0.9, 2.0)
        assert itr_high > itr_low

    def test_faster_trials_higher_itr(self):
        itr_slow = shannon_itr(8, 0.9, 5.0)
        itr_fast = shannon_itr(8, 0.9, 1.0)
        assert itr_fast > itr_slow

    def test_invalid_n_targets(self):
        with pytest.raises(ValueError):
            shannon_itr(1, 0.9, 2.0)


# ── Compute Episode Metrics ─────────────────────────────────────────

class TestComputeEpisodeMetrics:
    def test_successful_episode(self):
        positions = np.array([[0, 0], [0.1, 0], [0.2, 0], [0.3, 0], [0.4, 0]])
        actions = np.array([[1, 0], [1, 0], [1, 0], [1, 0]])
        m = compute_episode_metrics(
            cursor_positions=positions,
            actions=actions,
            start_pos=np.array([0, 0]),
            target_pos=np.array([0.4, 0]),
            target_radius=0.05,
            dt=0.1,
            success=True,
        )
        assert m.success is True
        assert m.steps == 4
        assert m.movement_time == pytest.approx(0.4)
        assert m.path_efficiency == pytest.approx(1.0, abs=1e-6)
        assert m.action_smoothness == pytest.approx(0.0)
        assert m.fitts_id > 0
        assert m.fitts_tp > 0

    def test_failed_episode(self):
        positions = np.array([[0, 0], [0.05, 0]])
        actions = np.array([[0.5, 0]])
        m = compute_episode_metrics(
            cursor_positions=positions,
            actions=actions,
            start_pos=np.array([0, 0]),
            target_pos=np.array([0.4, 0]),
            target_radius=0.05,
            dt=0.1,
            success=False,
        )
        assert m.success is False
        assert m.final_distance > 0


# ── Aggregate Metrics ────────────────────────────────────────────────

class TestAggregateMetrics:
    def _make_episode(self, success, steps, smoothness=0.1):
        return EpisodeMetrics(
            success=success,
            movement_time=steps * 0.1,
            steps=steps,
            fitts_id=2.0,
            fitts_tp=2.0 / (steps * 0.1),
            path_efficiency=0.8,
            action_smoothness=smoothness,
            mean_action_magnitude=0.5,
            final_distance=0.0 if success else 0.2,
            target_distance=0.4,
            cursor_trajectory=np.zeros((steps + 1, 2)),
            actions=np.zeros((steps, 2)),
        )

    def test_all_successful(self):
        episodes = [self._make_episode(True, 20), self._make_episode(True, 30)]
        agg = aggregate_metrics(episodes)
        assert agg["success_rate"] == 1.0
        assert agg["steps_mean"] == 25.0

    def test_mixed_success(self):
        episodes = [
            self._make_episode(True, 20),
            self._make_episode(False, 200),
        ]
        agg = aggregate_metrics(episodes)
        assert agg["success_rate"] == 0.5
        # Time-based metrics only from successful episodes
        assert agg["steps_mean"] == 20.0

    def test_empty(self):
        assert aggregate_metrics([]) == {}

    def test_has_ci(self):
        episodes = [self._make_episode(True, s) for s in [20, 25, 30, 22, 28]]
        agg = aggregate_metrics(episodes)
        assert "steps_ci95" in agg
        assert agg["steps_ci95"] > 0
