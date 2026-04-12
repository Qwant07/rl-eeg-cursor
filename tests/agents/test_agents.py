"""Tests for RL agents: ConstrainedRewardWrapper and BehaviorCloning."""
import numpy as np
import pytest

from src.envs.cursor_env import CursorEnv
from src.envs.noise_model import DecoderNoiseModel
from src.agents.constrained_wrapper import ConstrainedRewardWrapper
from src.agents.behavior_cloning import BehaviorCloning


# ── ConstrainedRewardWrapper ─────────────────────────────────────────

class TestConstrainedRewardWrapper:
    def test_base_reward_in_info(self):
        env = CursorEnv()
        wrapped = ConstrainedRewardWrapper(env, lambda_smooth=0.1, lambda_zero=0.05)
        wrapped.reset(seed=0)
        _, _, _, _, info = wrapped.step(np.array([0.5, 0.5]))
        assert "base_reward" in info
        assert "smooth_penalty" in info
        assert "zero_penalty" in info

    def test_zero_action_no_penalty(self):
        env = CursorEnv()
        wrapped = ConstrainedRewardWrapper(env, lambda_smooth=0.1, lambda_zero=0.0)
        wrapped.reset(seed=0)
        _, r1, _, _, info1 = wrapped.step(np.array([0.0, 0.0]))
        assert info1["zero_penalty"] == pytest.approx(0.0)
        assert info1["smooth_penalty"] == pytest.approx(0.0)

    def test_penalty_reduces_reward(self):
        env = CursorEnv()
        # Run without wrapper
        env.reset(seed=0)
        _, base_reward, _, _, _ = env.step(np.array([1.0, 1.0]))

        # Run with wrapper
        wrapped = ConstrainedRewardWrapper(
            CursorEnv(), lambda_smooth=0.0, lambda_zero=0.1,
        )
        wrapped.reset(seed=0)
        _, penalized_reward, _, _, info = wrapped.step(np.array([1.0, 1.0]))

        assert penalized_reward < base_reward
        assert info["zero_penalty"] > 0

    def test_smoothness_penalty_on_action_change(self):
        env = CursorEnv()
        wrapped = ConstrainedRewardWrapper(env, lambda_smooth=1.0, lambda_zero=0.0)
        wrapped.reset(seed=0)
        wrapped.step(np.array([1.0, 0.0]))
        _, _, _, _, info = wrapped.step(np.array([-1.0, 0.0]))
        # Change of 2.0 in x → penalty = 1.0 * (2.0² + 0²) = 4.0
        assert info["smooth_penalty"] == pytest.approx(4.0)

    def test_no_smooth_penalty_on_first_step(self):
        env = CursorEnv()
        wrapped = ConstrainedRewardWrapper(env, lambda_smooth=1.0, lambda_zero=0.0)
        wrapped.reset(seed=0)
        _, _, _, _, info = wrapped.step(np.array([1.0, 1.0]))
        assert info["smooth_penalty"] == pytest.approx(0.0)

    def test_reset_clears_prev_action(self):
        env = CursorEnv()
        wrapped = ConstrainedRewardWrapper(env, lambda_smooth=1.0, lambda_zero=0.0)
        wrapped.reset(seed=0)
        wrapped.step(np.array([1.0, 0.0]))
        wrapped.reset(seed=1)
        _, _, _, _, info = wrapped.step(np.array([1.0, 0.0]))
        assert info["smooth_penalty"] == pytest.approx(0.0)


# ── BehaviorCloning ──────────────────────────────────────────────────

class TestBehaviorCloning:
    def test_collect_and_train(self):
        env = CursorEnv()
        bc = BehaviorCloning(env)
        bc.collect(n_episodes=10)
        assert len(bc.obs_data) > 0
        losses = bc.train(epochs=3, batch_size=32)
        assert len(losses) == 3
        assert losses[-1] < losses[0]  # loss should decrease

    def test_predict_shape(self):
        env = CursorEnv()
        bc = BehaviorCloning(env)
        bc.collect(n_episodes=5)
        bc.train(epochs=2)
        obs, _ = env.reset(seed=0)
        action = bc.predict(obs)
        assert action.shape == (2,)
        assert np.all(np.abs(action) <= 1.0)  # tanh bounded

    def test_evaluate_returns_dict(self):
        env = CursorEnv()
        bc = BehaviorCloning(env)
        bc.collect(n_episodes=5)
        bc.train(epochs=2)
        result = bc.evaluate(n_episodes=5)
        assert "success_rate" in result
        assert "avg_steps" in result

    def test_save_load_roundtrip(self, tmp_path):
        env = CursorEnv()
        bc = BehaviorCloning(env)
        bc.collect(n_episodes=5)
        bc.train(epochs=2)

        path = str(tmp_path / "bc_policy.pt")
        bc.save(path)

        bc2 = BehaviorCloning(env)
        bc2.load(path)

        obs, _ = env.reset(seed=0)
        a1 = bc.predict(obs)
        a2 = bc2.predict(obs)
        np.testing.assert_allclose(a1, a2, atol=1e-6)

    def test_expert_action_points_toward_target(self):
        env = CursorEnv()
        bc = BehaviorCloning(env)
        obs, _ = env.reset(seed=0)
        action = bc._expert_action(obs)
        # Action should point from cursor toward target (in normalized coords)
        direction = obs[2:4] - obs[:2]
        dist = np.linalg.norm(direction)
        if dist > 0.01:
            expected = direction / dist
            np.testing.assert_allclose(action, expected, atol=0.01)
