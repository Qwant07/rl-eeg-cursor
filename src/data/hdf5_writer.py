import os
import h5py
import numpy as np


class HDF5Writer:
    """Context manager for writing preprocessed EEG epochs to an HDF5 file.

    File layout::

        /<subject>/session_<NN>/<run_type>/run_<NN>/
            X       — (N, n_ch, n_times) float32, compressed
            y       — (N, 2)             float32
            cursor  — (N, 2)             float32
            target  — (N, 2)             float32
            mean    — (n_ch,)            float64
            std     — (n_ch,)            float64

    Usage::

        with HDF5Writer("preprocessed/S01_preprocessed.h5") as writer:
            writer.write_run("S01", 1, "AR", 1, X, y, cursor, target, mean, std)
    """

    def __init__(self, filepath: str):
        self.filepath = filepath
        self._file: h5py.File | None = None

    def __enter__(self) -> "HDF5Writer":
        os.makedirs(os.path.dirname(self.filepath) or ".", exist_ok=True)
        self._file = h5py.File(self.filepath, "a")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._file is not None:
            self._file.close()
            self._file = None

    def write_run(
        self,
        subject: str,
        session: int,
        run_type: str,
        run_num: int,
        X: np.ndarray,
        y: np.ndarray,
        cursor: np.ndarray,
        target: np.ndarray,
        mean: np.ndarray,
        std: np.ndarray,
    ) -> None:
        """Write one preprocessed run into the HDF5 file.

        Args:
            subject:  Subject identifier, e.g. "S01"
            session:  Session number (1-indexed)
            run_type: Run type string, e.g. "AR", "EG", "PN"
            run_num:  Run number within session (1-indexed)
            X:        (N, n_ch, n_times) EEG epochs
            y:        (N, 2) cursor velocity labels
            cursor:   (N, 2) cursor positions
            target:   (N, 2) target positions
            mean:     (n_ch,) per-channel mean used for z-scoring
            std:      (n_ch,) per-channel std used for z-scoring
        """
        assert self._file is not None, "HDF5Writer must be used as a context manager"

        key = f"{subject}/session_{session:02d}/{run_type}/run_{run_num:02d}"
        grp = self._file.require_group(key)

        for name, arr, dtype in [
            ("X",      X,      np.float32),
            ("y",      y,      np.float32),
            ("cursor", cursor, np.float32),
            ("target", target, np.float32),
            ("mean",   mean,   np.float64),
            ("std",    std,    np.float64),
        ]:
            if name in grp:
                del grp[name]
            grp.create_dataset(
                name,
                data=arr.astype(dtype),
                compression="gzip",
                compression_opts=4,
            )
