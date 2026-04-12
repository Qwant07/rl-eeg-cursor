"""Tests for CursorEnv and DecoderNoiseModel."""
import numpy as np
import pytest

from src.envs.cursor_env import CursorEnv
from src.envs.noise_model import DecoderNoiseModel


# ── DecoderNoiseModel ────────────────────────────────────────────────

class TestDecoderNoiseModel:
    def test_identity_without_noise(self):
        """Zero noise, unit gain → output equals input."""
        model = DecoderNoiseModel(
            gain=np.array([1.0, 1.0]),
            bias=np.array([0.0, 0.0]),
            noise_cov=np.zeros((2, 2)),
        )
        vel = np.array([0.3, -0.2])
        out = model(vel)
        np.testing.assert_allclose(out, vel, atol=1e-7)

    def test_gain_attenuation(self):
        model = DecoderNoiseModel(
            gain=np.array([0.5, 0.5]),
            bias=np.array([0.0, 0.0]),
            noise_cov=np.zeros((2, 2)),
        )
        vel = np.array([1.0, 1.0])
        out = model(vel)
        np.testing.assert_allclose(out, [0.5, 0.5], atol=1e-7)

    def test_bias(self):
        model = DecoderNoiseModel(
            gain=np.array([1.0, 1.0]),
            bias=np.array([0.1, -0.1]),
            noise_cov=np.zeros((2, 2)),
        )
        out = model(np.array([0.0, 0.0]))
        np.testing.assert_allclose(out, [0.1, -0.1], atol=1e-7)

    def test_noise_has_correct_statistics(self):
        rng = np.random.default_rng(42)
        cov = np.array([[0.04, 0.01], [0.01, 0.09]])
        model = DecoderNoiseModel(
            gain=np.array([1.0, 1.0]),
            bias=np.array([0.0, 0.0]),
            noise_cov=cov,
            rng=rng,
        )
        samples = np.array([model(np.zeros(2)) for _ in range(10000)])
        empirical_cov = np.cov(samples.T)
        np.testing.assert_allclose(empirical_cov, cov, atol=0.01)

    def test_from_subject_s01(self):
        model = DecoderNoiseModel.from_subject("S01")
        assert model.gain.shape == (2,)
        assert model.gain[0] < 0.5  # S01 has weak gain

    def test_from_subject_s05(self):
        model = DecoderNoiseModel.from_subject("S05")
        assert model.gain[1] > 0.5  # S05 has stronger gain

    def test_from_subject_invalid(self):
        with pytest.raises(ValueError, match="Unknown subject"):
            DecoderNoiseModel.from_subject("S99")

    def test_serialization_roundtrip(self):
        model = DecoderNoiseModel.from_subject("S05", rng=np.random.default_rng(0))
        d = model.to_dict()
        restored = DecoderNoiseModel.from_dict(d, rng=np.random.default_rng(0))
        np.testing.assert_allclose(restored.gain, model.gain)
        np.testing.assert_allclose(restored.bias, model.bias)
        np.testing.assert_allclose(restored.noise_cov, model.noise_cov)


# ── CursorEnv ────────────────────────────────────────────────────────

class TestCursorEnv:
    def test_reset_returns_valid_obs(self):
        env = CursorEnv()
        obs, info = env.reset(seed=42)
        assert obs.shape == (7,)
        assert env.observation_space.contains(obs)
        assert info["distance"] > 0

    def test_step_returns_valid(self):
        env = CursorEnv()
        env.reset(seed=42)
        action = np.array([0.5, 0.5], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        assert obs.shape == (7,)
        assert isinstance(reward, float)
        assert not terminated  # unlikely to reach target in 1 step

    def test_cursor_moves_toward_action(self):
        """With no noise model, cursor should move in action direction."""
        env = CursorEnv(noise_model=None, dt=0.1, vel_scale=1.0)
        env.reset(seed=42)
        env.cursor_pos = np.array([0.0, 0.0], dtype=np.float32)
        action = np.array([1.0, 0.0], dtype=np.float32)
        env.step(action)
        assert env.cursor_pos[0] > 0.0  # moved right
        assert abs(env.cursor_pos[1]) < 1e-6  # didn't move vertically

    def test_cursor_clipped_to_workspace(self):
        env = CursorEnv(noise_model=None, dt=1.0, vel_scale=10.0)
        env.reset(seed=42)
        env.step(np.array([1.0, 1.0]))
        assert env.cursor_pos[0] <= env.workspace
        assert env.cursor_pos[1] <= env.workspace

    def test_episode_terminates_on_dwell(self):
        """Cursor at target for dwell_steps → terminated."""
        env = CursorEnv(noise_model=None, dwell_steps=2, target_radius=1.0)
        env.reset(seed=42)
        # Target radius is huge, so cursor is already inside
        _, _, t1, _, _ = env.step(np.array([0.0, 0.0]))
        _, _, t2, _, _ = env.step(np.array([0.0, 0.0]))
        assert t2  # should terminate after 2 dwells

    def test_episode_truncates_at_max_steps(self):
        env = CursorEnv(noise_model=None, max_steps=5)
        env.reset(seed=42)
        for _ in range(4):
            _, _, terminated, truncated, _ = env.step(np.array([0.0, 0.0]))
            assert not truncated
        _, _, terminated, truncated, _ = env.step(np.array([0.0, 0.0]))
        assert truncated

    def test_success_bonus_in_reward(self):
        env = CursorEnv(
            noise_model=None, dwell_steps=1,
            target_radius=1.0, success_bonus=10.0,
        )
        env.reset(seed=42)
        _, reward, terminated, _, _ = env.step(np.array([0.0, 0.0]))
        assert terminated
        assert reward > 5.0  # includes bonus

    def test_negative_reward_when_far(self):
        env = CursorEnv(noise_model=None, target_radius=0.01)
        env.reset(seed=42)
        _, reward, _, _, _ = env.step(np.array([0.0, 0.0]))
        assert reward < 0  # negative distance

    def test_with_noise_model(self):
        noise = DecoderNoiseModel.from_subject("S05", rng=np.random.default_rng(0))
        env = CursorEnv(noise_model=noise)
        obs, _ = env.reset(seed=42)
        obs2, reward, _, _, _ = env.step(np.array([1.0, 0.0]))
        assert obs2.shape == (7,)

    def test_latency_delays_velocity(self):
        """With latency, first step should use zero velocity (from buffer init)."""
        env = CursorEnv(noise_model=None, latency_steps=2, dt=0.1, vel_scale=1.0)
        env.reset(seed=42)
        start_pos = env.cursor_pos.copy()
        env.step(np.array([1.0, 0.0]))  # buffered, not applied yet
        # With latency=2, first two steps use the zero-init buffer
        np.testing.assert_allclose(env.cursor_pos, start_pos, atol=1e-6)

    def test_eight_targets_default(self):
        env = CursorEnv()
        assert len(env.target_positions) == 8

    def test_custom_targets(self):
        targets = [[0.3, 0.0], [-0.3, 0.0]]
        env = CursorEnv(target_positions=targets)
        assert len(env.target_positions) == 2

    def test_seed_reproducibility(self):
        env = CursorEnv()
        obs1, _ = env.reset(seed=123)
        obs2, _ = env.reset(seed=123)
        np.testing.assert_array_equal(obs1, obs2)

    def test_time_remaining_decreases(self):
        env = CursorEnv(max_steps=10)
        obs, _ = env.reset(seed=42)
        assert obs[6] == pytest.approx(1.0)
        obs, _, _, _, _ = env.step(np.array([0.0, 0.0]))
        assert obs[6] == pytest.approx(0.9, abs=0.01)

    def test_gymnasium_api_check(self):
        """Verify env passes gymnasium's API checker."""
        from gymnasium.utils.env_checker import check_env
        env = CursorEnv()
        check_env(env.unwrapped, skip_render_check=True)
