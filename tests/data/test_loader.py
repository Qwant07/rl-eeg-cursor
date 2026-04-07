import numpy as np
import pytest
from src.data.loader import load_run

MAT_PATH = "project_cursor/data/S01/S01_Se01_AR_R01.mat"


def test_load_run_shapes():
    eeg, cursor_vel, cursor_pos, target_pos, eeg_times, pos_times, fs = load_run(MAT_PATH)
    assert eeg.shape == (317280, 62), f"eeg shape: {eeg.shape}"
    assert cursor_vel.shape[1] == 2
    assert cursor_pos.shape[1] == 2
    assert target_pos.shape[1] == 2
    assert cursor_vel.shape[0] == cursor_pos.shape[0] == target_pos.shape[0] == len(pos_times)
    assert len(eeg_times) == 317280


def test_load_run_fs():
    _, _, _, _, _, _, fs = load_run(MAT_PATH)
    assert fs == 1000.0


def test_load_run_dtypes():
    eeg, cursor_vel, cursor_pos, target_pos, eeg_times, pos_times, fs = load_run(MAT_PATH)
    assert eeg.dtype == np.float64
    assert cursor_vel.dtype == np.float64
