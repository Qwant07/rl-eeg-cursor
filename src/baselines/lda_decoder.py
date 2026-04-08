"""Band-power LDA/Ridge baseline decoder for continuous 2-D cursor velocity.

BandPowerLDA extracts log-band-power features from EEG epochs and fits a
Ridge regression model (one per velocity axis) to predict cursor velocity.
It is referred to as "LDA" because linear discriminant / linear regression on
band-power is the standard classical BCI baseline.

Bands (Hz): delta 1-4, theta 4-8, alpha 8-13, beta 13-30, gamma 30-40
Features per epoch: n_channels × 5 bands
"""
from typing import Tuple, Dict

import numpy as np
from scipy.signal import welch
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score


_BANDS = {
    "delta": (1, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta":  (13, 30),
    "gamma": (30, 40),
}


def _band_power(epoch: np.ndarray, fs: float) -> np.ndarray:
    """Compute log-band-power features for a single epoch.

    Args:
        epoch: (n_ch, n_times) EEG epoch
        fs:    sampling frequency in Hz

    Returns:
        features: (n_ch * n_bands,) float64
    """
    n_ch, n_times = epoch.shape
    # Welch PSD — nperseg capped to epoch length
    nperseg = min(256, n_times)
    freqs, psd = welch(epoch, fs=fs, nperseg=nperseg, axis=-1)  # (n_ch, n_freqs)

    feats = []
    for lo, hi in _BANDS.values():
        mask = (freqs >= lo) & (freqs < hi)
        band_p = psd[:, mask].mean(axis=1)          # (n_ch,)
        feats.append(np.log1p(band_p))               # log-power, stable near 0
    return np.concatenate(feats)                     # (n_ch * n_bands,)


def extract_features(X: np.ndarray, fs: float) -> np.ndarray:
    """Extract log-band-power features from a batch of epochs.

    Args:
        X:  (N, n_ch, n_times) EEG epochs
        fs: sampling frequency in Hz

    Returns:
        F: (N, n_ch * n_bands) feature matrix
    """
    return np.array([_band_power(X[i], fs) for i in range(len(X))])


class BandPowerLDA:
    """Linear baseline: log-band-power → Ridge regression → 2-D velocity.

    Args:
        fs:    EEG sampling frequency in Hz (default 1000)
        alpha: Ridge regularisation strength (default 1.0)
    """

    def __init__(self, fs: float = 1000.0, alpha: float = 1.0):
        self.fs = fs
        self.alpha = alpha
        self._models = [Ridge(alpha=alpha), Ridge(alpha=alpha)]  # vx, vy
        self._fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "BandPowerLDA":
        """Fit decoder on training epochs.

        Args:
            X: (N, n_ch, n_times) training EEG epochs
            y: (N, 2) cursor velocity labels (vx, vy)
        """
        F = extract_features(X, self.fs)
        self._models[0].fit(F, y[:, 0])
        self._models[1].fit(F, y[:, 1])
        self._fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict 2-D cursor velocity.

        Args:
            X: (N, n_ch, n_times) EEG epochs

        Returns:
            y_hat: (N, 2) predicted velocity (vx, vy)
        """
        assert self._fitted, "Call fit() before predict()"
        F = extract_features(X, self.fs)
        vx = self._models[0].predict(F)
        vy = self._models[1].predict(F)
        return np.stack([vx, vy], axis=1)

    def score(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Compute NMSE and R² for both velocity axes.

        NMSE = MSE(y_hat, y) / Var(y)

        Args:
            X: (N, n_ch, n_times) EEG epochs
            y: (N, 2) ground-truth velocity

        Returns:
            dict with keys: nmse_vx, nmse_vy, nmse_mean, r2_vx, r2_vy, r2_mean
        """
        y_hat = self.predict(X)
        results = {}
        for i, axis in enumerate(["vx", "vy"]):
            mse = np.mean((y_hat[:, i] - y[:, i]) ** 2)
            var = np.var(y[:, i])
            nmse = mse / var if var > 0 else float("inf")
            r2 = float(r2_score(y[:, i], y_hat[:, i]))
            results[f"nmse_{axis}"] = float(nmse)
            results[f"r2_{axis}"] = r2
        results["nmse_mean"] = (results["nmse_vx"] + results["nmse_vy"]) / 2
        results["r2_mean"]   = (results["r2_vx"]   + results["r2_vy"])   / 2
        return results
