"""Tests for epoch_run and zscore_normalize."""
import numpy as np
import pytest
from src.data.preprocessor import filter_raw, epoch_run, zscore_normalize

FS = 1000.0
N_SAMPLES = 5000   # 5 s of EEG
N_CH = 62
WINDOW_MS = 500.0
STRIDE_MS = 250.0
N_POS = 125        # 5 s at 25 Hz


def _make_inputs():
    rng = np.random.default_rng(0)
    eeg = rng.standard_normal((N_SAMPLES, N_CH))
    raw = filter_raw(eeg, FS)

    eeg_times = np.arange(N_SAMPLES, dtype=np.float64)
    # pos_times at 25 Hz: every 40 EEG samples
    pos_times = np.arange(N_POS, dtype=np.float64) * 40.0
    cursor_vel = rng.standard_normal((N_POS, 2))
    cursor_pos = rng.standard_normal((N_POS, 2))
    target_pos = rng.standard_normal((N_POS, 2))
    return raw, eeg_times, cursor_vel, cursor_pos, target_pos, pos_times


def test_epoch_run_output_shapes():
    raw, eeg_times, cursor_vel, cursor_pos, target_pos, pos_times = _make_inputs()
    X, y, cursor, target = epoch_run(
        raw, eeg_times, cursor_vel, cursor_pos, target_pos, pos_times,
        window_ms=WINDOW_MS, stride_ms=STRIDE_MS,
    )
    window_samples = int(WINDOW_MS * FS / 1000)
    assert X.ndim == 3
    assert X.shape[1] == N_CH
    assert X.shape[2] == window_samples
    assert y.shape == (X.shape[0], 2)
    assert cursor.shape == (X.shape[0], 2)
    assert target.shape == (X.shape[0], 2)


def test_epoch_run_epoch_count():
    raw, eeg_times, cursor_vel, cursor_pos, target_pos, pos_times = _make_inputs()
    X, y, _, _ = epoch_run(
        raw, eeg_times, cursor_vel, cursor_pos, target_pos, pos_times,
        window_ms=WINDOW_MS, stride_ms=STRIDE_MS,
    )
    # Expected: floor((N_SAMPLES - window_samples) / stride_samples) + 1
    window_samples = int(WINDOW_MS * FS / 1000)
    stride_samples = int(STRIDE_MS * FS / 1000)
    expected_n = (N_SAMPLES - window_samples) // stride_samples + 1
    assert X.shape[0] == expected_n


def test_epoch_run_dtypes():
    raw, eeg_times, cursor_vel, cursor_pos, target_pos, pos_times = _make_inputs()
    X, y, cursor, target = epoch_run(
        raw, eeg_times, cursor_vel, cursor_pos, target_pos, pos_times,
    )
    for arr in (X, y, cursor, target):
        assert arr.dtype == np.float64


def test_zscore_normalize_shape():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((100, N_CH, 500))
    X_norm, mean, std = zscore_normalize(X)
    assert X_norm.shape == X.shape
    assert mean.shape == (N_CH,)
    assert std.shape == (N_CH,)


def test_zscore_normalize_zero_mean_unit_std():
    rng = np.random.default_rng(2)
    X = rng.standard_normal((200, N_CH, 500))
    X_norm, mean, std = zscore_normalize(X)
    # After normalisation the global per-channel mean should be ~0
    flat = X_norm.reshape(X_norm.shape[0], N_CH, -1)
    channel_means = flat.mean(axis=(0, 2))
    channel_stds  = flat.std(axis=(0, 2))
    np.testing.assert_allclose(channel_means, 0.0, atol=1e-10)
    np.testing.assert_allclose(channel_stds,  1.0, atol=1e-10)


def test_zscore_normalize_constant_channel():
    """Constant channel should not produce NaN/inf."""
    X = np.ones((10, N_CH, 500))
    X_norm, mean, std = zscore_normalize(X)
    assert np.all(np.isfinite(X_norm))
