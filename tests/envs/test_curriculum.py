"""Tests for CurriculumWrapper."""
import numpy as np
import pytest

from src.envs.cursor_env import CursorEnv
from src.envs.curriculum import CurriculumWrapper


class TestCurriculumWrapper:
    def test_initial_radius_is_enlarged(self):
        env = CursorEnv()
        base_radius = env.target_radius
        wrapped = CurriculumWrapper(env, total_steps=1000, initial_radius_mult=4.0)
        wrapped.reset(seed=0)
        assert env.target_radius == pytest.approx(base_radius * 4.0)

    def test_final_radius_is_normal(self):
        env = CursorEnv()
        base_radius = env.target_radius
        wrapped = CurriculumWrapper(env, total_steps=100, end_frac=0.5)
        wrapped.reset(seed=0)
        # Step past end_frac
        for _ in range(60):
            wrapped.step(np.zeros(2, dtype=np.float32))
        wrapped.reset(seed=1)
        assert env.target_radius == pytest.approx(base_radius, abs=0.001)

    def test_progress_starts_at_zero(self):
        env = CursorEnv()
        wrapped = CurriculumWrapper(env, total_steps=1000)
        assert wrapped.progress == 0.0

    def test_progress_reaches_one(self):
        env = CursorEnv()
        wrapped = CurriculumWrapper(env, total_steps=100, end_frac=0.5)
        wrapped._global_step = 100
        assert wrapped.progress == 1.0

    def test_targets_start_close(self):
        env = CursorEnv()
        base_dist = np.linalg.norm(env.target_positions[0])
        wrapped = CurriculumWrapper(env, total_steps=1000, initial_distance_mult=0.3)
        wrapped.reset(seed=0)
        current_dist = np.linalg.norm(env.target_positions[0])
        assert current_dist == pytest.approx(base_dist * 0.3, abs=0.01)

    def test_info_contains_curriculum_progress(self):
        env = CursorEnv()
        wrapped = CurriculumWrapper(env, total_steps=1000)
        _, info = wrapped.reset(seed=0)
        assert "curriculum_progress" in info
        _, _, _, _, info = wrapped.step(np.zeros(2, dtype=np.float32))
        assert "curriculum_progress" in info

    def test_step_increments_global_step(self):
        env = CursorEnv()
        wrapped = CurriculumWrapper(env, total_steps=1000)
        wrapped.reset(seed=0)
        wrapped.step(np.zeros(2, dtype=np.float32))
        assert wrapped._global_step == 1
        wrapped.step(np.zeros(2, dtype=np.float32))
        assert wrapped._global_step == 2

    def test_gymnasium_api_compliance(self):
        from gymnasium.utils.env_checker import check_env
        env = CursorEnv()
        wrapped = CurriculumWrapper(env, total_steps=1000)
        check_env(wrapped.unwrapped, skip_render_check=True)
