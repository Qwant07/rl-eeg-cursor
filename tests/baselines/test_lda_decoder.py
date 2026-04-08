"""Tests for BandPowerLDA baseline decoder."""
import numpy as np
import pytest
from src.baselines.lda_decoder import BandPowerLDA, extract_features

FS = 1000.0
N_TRAIN = 200
N_TEST = 50
N_CH = 62
N_T = 500   # 500 ms @ 1000 Hz


def _make_data(n, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, N_CH, N_T))
    y = rng.standard_normal((n, 2))
    return X, y


def test_extract_features_shape():
    X, _ = _make_data(10)
    F = extract_features(X, FS)
    assert F.shape == (10, N_CH * 5)


def test_extract_features_finite():
    X, _ = _make_data(10)
    F = extract_features(X, FS)
    assert np.all(np.isfinite(F))


def test_bandpowerlda_fit_predict_shapes():
    X_tr, y_tr = _make_data(N_TRAIN, seed=1)
    X_te, _    = _make_data(N_TEST,  seed=2)
    model = BandPowerLDA(fs=FS)
    model.fit(X_tr, y_tr)
    y_hat = model.predict(X_te)
    assert y_hat.shape == (N_TEST, 2)


def test_bandpowerlda_predict_requires_fit():
    X, _ = _make_data(10)
    model = BandPowerLDA(fs=FS)
    with pytest.raises(AssertionError):
        model.predict(X)


def test_bandpowerlda_score_keys():
    X_tr, y_tr = _make_data(N_TRAIN, seed=3)
    X_te, y_te = _make_data(N_TEST,  seed=4)
    model = BandPowerLDA(fs=FS).fit(X_tr, y_tr)
    scores = model.score(X_te, y_te)
    for key in ("nmse_vx", "nmse_vy", "nmse_mean", "r2_vx", "r2_vy", "r2_mean"):
        assert key in scores, f"Missing key: {key}"


def test_bandpowerlda_score_finite():
    X_tr, y_tr = _make_data(N_TRAIN, seed=5)
    X_te, y_te = _make_data(N_TEST,  seed=6)
    model = BandPowerLDA(fs=FS).fit(X_tr, y_tr)
    scores = model.score(X_te, y_te)
    for k, v in scores.items():
        assert np.isfinite(v), f"{k} = {v} is not finite"


def test_bandpowerlda_fits_perfectly_on_trivial_data():
    """When y is perfectly predictable from features, NMSE should be very low."""
    rng = np.random.default_rng(99)
    X = rng.standard_normal((300, N_CH, N_T))
    F = extract_features(X, FS)
    # Make y a linear function of the first two features
    y = np.column_stack([F[:, 0], F[:, 1]])

    model = BandPowerLDA(fs=FS, alpha=1e-6).fit(X, y)
    scores = model.score(X, y)
    assert scores["nmse_mean"] < 0.05, f"NMSE too high: {scores['nmse_mean']:.4f}"
    assert scores["r2_mean"] > 0.95,   f"R² too low: {scores['r2_mean']:.4f}"
