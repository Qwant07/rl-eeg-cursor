---
tags: [memory, progress, week1]
---

# Week 1 Progress

## Status: COMPLETE

## Completed
- [x] Task 0: Project setup
  - Git repo initialized at `project_cursor/`
  - Directory structure: `src/data/`, `src/baselines/`, `tests/data/`, `tests/baselines/`, `preprocessed/`
  - `requirements.txt`: h5py, mne, sklearn, numpy, scipy, matplotlib, torch, pytest
  - MNE 1.11.0 installed and verified
  - `.gitignore`: `/data/`, `preprocessed/*.h5`, `.DS_Store`, `.venv/`, etc.

- [x] Task 1: MAT file loader (`src/data/loader.py`)
  - `load_run(path)` using `h5py` (NOT scipy — files are HDF5 v7.3)
  - Returns: `(eeg, cursor_vel, cursor_pos, target_pos, eeg_times, pos_times, fs)`
  - 3 tests passing: shape, fs=1000.0, dtypes all float64
  - Fix applied: robust `fs` read via `np.asarray().flat[0]`

- [x] Task 2: MNE filter pipeline (`src/data/preprocessor.py` — `filter_raw`)
  - Bandpass 1–40 Hz (4th-order Butterworth) + notch 50 Hz + CAR
  - Returns `mne.io.RawArray` — internal shape `(62, n_samples)`
  - 3 tests passing: return type, shape preserved, 80 Hz attenuation

- [x] Task 3: Epoching + z-score (`epoch_run`, `zscore_normalize` in `preprocessor.py`)
  - `epoch_run(raw, eeg_times, cursor_vel, cursor_pos, target_pos, pos_times, window_ms=500, stride_ms=250)`
  - Aligns EEG windows to nearest pos_times sample at window end
  - Returns `(X, y, cursor, target)` — X shape `(N, 62, 500)` at 1000 Hz
  - `zscore_normalize(X)` → `(X_norm, mean, std)` per-channel across all epochs+time
  - 6 tests passing: shapes, epoch count, dtypes, z-score properties, constant channel safety

- [x] Task 4: HDF5 writer (`src/data/hdf5_writer.py`)
  - `HDF5Writer` context manager with `write_run(subject, session, run_type, run_num, X, y, cursor, target, mean, std)`
  - HDF5 layout: `/<subject>/session_<NN>/<run_type>/run_<NN>/`
  - Datasets: X (float32, gzip-4), y, cursor, target (float32), mean, std (float64)
  - 5 tests passing: creates file, shapes, multiple runs, overwrite, value roundtrip

- [x] Task 5: Main preprocessing script (`src/preprocess.py`)
  - `python -m src.preprocess --subject S01 [--data_dir data] [--out_dir preprocessed]`
  - Parses `S01_Se01_AR_R01.mat` filename convention
  - Full pipeline: load → filter → epoch → zscore → write HDF5
  - Ready to run on S01/S05 once data is available

- [x] Task 6: LDA baseline decoder (`src/baselines/lda_decoder.py`)
  - `BandPowerLDA`: log-band-power features (5 bands × 62 ch = 310 features) + Ridge regression
  - Bands: delta 1-4, theta 4-8, alpha 8-13, beta 13-30, gamma 30-40 Hz
  - `fit(X, y)`, `predict(X)`, `score(X, y)` → NMSE and R² per axis + mean
  - 7 tests passing including perfect-fit sanity test

## Test Summary
- **21 / 21 tests passing** across all Week 1 modules (36 total with Week 2)
- Files: `tests/data/test_loader.py`, `test_preprocessor.py`, `test_preprocessor_epoch.py`,
  `test_hdf5_writer.py`, `tests/baselines/test_lda_decoder.py`

## Deliverable Status
- `src/preprocess.py` ready — ✅ ran successfully on both subjects (2026-04-08)
- `preprocessed/S01_preprocessed.h5` — ✅ 6.8 GB, 104 runs, ~1268 epochs/run
- `preprocessed/S05_preprocessed.h5` — ✅ 15 GB, 104 runs, ~1268 epochs/run
- LDA NMSE + R² — pending (needs training loop on HDF5 data)
- HDF5 files ready to upload to Google Drive for Week 2

## Note on Environment
- Python 3.14.3 system install on Linux
- Installed system-wide via `pip --break-system-packages`: mne 1.11.0, h5py 3.16.0
- scikit-learn (sklearn-compat 0.1.5), scipy 1.17.1 already present

## GitHub
- Repo: https://github.com/Qwant07/rl-eeg-cursor
- Codex CLI installed (v0.118.0, authenticated)

## Next: Week 2
- Implement EEGNet decoder (`src/decoders/eegnet.py`) — Colab
- Implement LSTM decoder (`src/decoders/lstm.py`) — Colab
- Training loop with MSE + L2, Adam, cosine decay, early stopping
- Upload S01/S05 HDF5 to Google Drive first

## Links
- [[BCI Project Index]]
- [[Architecture Decisions]]
