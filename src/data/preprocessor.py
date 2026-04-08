import numpy as np
import mne
from typing import Tuple


def filter_raw(eeg: np.ndarray, fs: float) -> mne.io.RawArray:
    """Apply bandpass, notch, and common-average reference to raw EEG.

    Args:
        eeg:  (n_samples, n_channels) float64
        fs:   sampling frequency in Hz

    Returns:
        mne.io.RawArray with shape (n_channels, n_samples), filtered in-place
    """
    n_samples, n_ch = eeg.shape
    ch_names = [f'EEG{i:03d}' for i in range(n_ch)]
    info = mne.create_info(ch_names=ch_names, sfreq=fs, ch_types='eeg')

    # MNE expects (n_channels, n_samples)
    raw = mne.io.RawArray(eeg.T.copy(), info, verbose=False)

    # Bandpass 1-40 Hz (4th-order Butterworth)
    raw.filter(
        l_freq=1.0, h_freq=40.0,
        method='iir',
        iir_params={'order': 4, 'ftype': 'butter'},
        verbose=False,
    )

    # Notch at 50 Hz (power line)
    raw.notch_filter(freqs=50.0, verbose=False)

    # Common average reference
    raw.set_eeg_reference('average', projection=False, verbose=False)

    return raw


def epoch_run(
    raw: mne.io.BaseRaw,
    eeg_times: np.ndarray,
    cursor_vel: np.ndarray,
    cursor_pos: np.ndarray,
    target_pos: np.ndarray,
    pos_times: np.ndarray,
    window_ms: float = 500.0,
    stride_ms: float = 250.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Slide a window over the filtered EEG and extract aligned labels.

    Args:
        raw:        Filtered MNE RawArray (n_ch, n_samples)
        eeg_times:  (n_samples,) EEG sample indices
        cursor_vel: (n_pos, 2) cursor velocity at 25 Hz
        cursor_pos: (n_pos, 2) cursor position at 25 Hz
        target_pos: (n_pos, 2) target position at 25 Hz
        pos_times:  (n_pos,) EEG-aligned sample indices for position data
        window_ms:  epoch length in milliseconds (default 500)
        stride_ms:  stride between epochs in milliseconds (default 250)

    Returns:
        X:      (N, n_ch, window_samples) float64 — EEG epochs
        y:      (N, 2) float64 — cursor velocity label at epoch end
        cursor: (N, 2) float64 — cursor position at epoch end
        target: (N, 2) float64 — target position at epoch end
    """
    data = raw.get_data()  # (n_ch, n_samples)
    fs = raw.info['sfreq']
    window_samples = int(window_ms * fs / 1000.0)
    stride_samples = int(stride_ms * fs / 1000.0)
    n_samples = data.shape[1]

    X_list, y_list, cursor_list, target_list = [], [], [], []

    start = 0
    while start + window_samples <= n_samples:
        end = start + window_samples
        epoch = data[:, start:end]  # (n_ch, window_samples)

        # Align to closest position sample at the end of the epoch
        end_time = eeg_times[end - 1]
        j = int(np.argmin(np.abs(pos_times - end_time)))

        X_list.append(epoch)
        y_list.append(cursor_vel[j])
        cursor_list.append(cursor_pos[j])
        target_list.append(target_pos[j])

        start += stride_samples

    X = np.array(X_list, dtype=np.float64)       # (N, n_ch, window_samples)
    y = np.array(y_list, dtype=np.float64)        # (N, 2)
    cursor = np.array(cursor_list, dtype=np.float64)  # (N, 2)
    target = np.array(target_list, dtype=np.float64)  # (N, 2)

    return X, y, cursor, target


def zscore_normalize(
    X: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Z-score normalize EEG epochs per channel across all epochs and time.

    Args:
        X: (N, n_ch, n_times) float64

    Returns:
        X_norm: (N, n_ch, n_times) normalized
        mean:   (n_ch,) per-channel mean
        std:    (n_ch,) per-channel standard deviation
    """
    # Collapse epoch and time dims, compute per-channel stats
    n, n_ch, n_t = X.shape
    flat = X.reshape(n, n_ch, -1)          # (N, n_ch, n_times)
    mean = flat.mean(axis=(0, 2))          # (n_ch,)
    std = flat.std(axis=(0, 2))            # (n_ch,)
    std = np.where(std == 0.0, 1.0, std)   # avoid division by zero

    X_norm = (X - mean[np.newaxis, :, np.newaxis]) / std[np.newaxis, :, np.newaxis]
    return X_norm, mean, std
