import numpy as np
import pytest
from src.data.preprocessor import filter_raw

N_SAMPLES = 10000
N_CH = 62
FS = 1000.0


def make_synthetic_eeg(n_samples=N_SAMPLES, n_ch=N_CH):
    rng = np.random.default_rng(42)
    return rng.standard_normal((n_samples, n_ch))


def test_filter_raw_returns_mne_raw():
    import mne
    eeg = make_synthetic_eeg()
    raw = filter_raw(eeg, FS)
    assert isinstance(raw, mne.io.BaseRaw)


def test_filter_raw_shape_preserved():
    eeg = make_synthetic_eeg()
    raw = filter_raw(eeg, FS)
    data = raw.get_data()  # MNE stores as (n_ch, n_samples)
    assert data.shape == (N_CH, N_SAMPLES)


def test_filter_raw_attenuates_high_freq():
    """Signal above 40 Hz should be attenuated."""
    t = np.linspace(0, N_SAMPLES / FS, N_SAMPLES)
    high_freq = np.tile(np.sin(2 * np.pi * 80 * t)[:, np.newaxis], (1, N_CH))
    raw = filter_raw(high_freq, FS)
    data = raw.get_data()
    assert np.std(data) < 0.1, "80 Hz signal should be nearly gone after 40 Hz lowpass"
