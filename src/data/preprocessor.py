import numpy as np
import mne


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
