import h5py
import numpy as np


def load_run(mat_path: str):
    """Load a single .mat (HDF5 v7.3) BCI run file.

    Args:
        mat_path: path to .mat file

    Returns:
        eeg:        (n_samples, 62) float64 — raw EEG, microvolts
        cursor_vel: (n_pos, 2)     float64 — (vx, vy) cursor velocity
        cursor_pos: (n_pos, 2)     float64 — (x, y) cursor position
        target_pos: (n_pos, 2)     float64 — (x, y) target position
        eeg_times:  (n_samples,)   float64 — sample indices for EEG
        pos_times:  (n_pos,)       float64 — sample indices for cursor/target
        fs:         float          — sampling frequency in Hz (1000.0)
    """
    with h5py.File(mat_path, 'r') as f:
        eeg = f['eeg/data'][:]                    # (n_samples, 62)
        fs = float(np.asarray(f['eeg/fs']).flat[0])
        eeg_times = f['eeg/times'][:, 0]          # (n_samples,)
        cursor_vel_x = f['eeg/cursorvel/x'][:, 0]
        cursor_vel_y = f['eeg/cursorvel/y'][:, 0]
        cursor_pos_x = f['eeg/cursorpos/x'][:, 0]
        cursor_pos_y = f['eeg/cursorpos/y'][:, 0]
        target_pos_x = f['eeg/targetpos/x'][:, 0]
        target_pos_y = f['eeg/targetpos/y'][:, 0]
        pos_times = f['eeg/postimes'][:, 0]        # (n_pos,)

    cursor_vel = np.stack([cursor_vel_x, cursor_vel_y], axis=1)
    cursor_pos = np.stack([cursor_pos_x, cursor_pos_y], axis=1)
    target_pos = np.stack([target_pos_x, target_pos_y], axis=1)

    return eeg, cursor_vel, cursor_pos, target_pos, eeg_times, pos_times, fs
