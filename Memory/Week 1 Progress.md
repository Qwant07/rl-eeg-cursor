---
tags: [memory, progress, week1]
---

# Week 1 Progress

## Status: In Progress

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

## Pending
- [ ] Task 3: Epoching + z-score (`epoch_run`, `zscore_normalize` in `preprocessor.py`)
- [ ] Task 4: HDF5 writer (`src/data/hdf5_writer.py`)
- [ ] Task 5: Main preprocessing script (`src/preprocess.py`) + run on S01/S05
- [ ] Task 6: LDA baseline decoder (`src/baselines/lda_decoder.py`)

## Deliverable Target
- `preprocessed/S01_preprocessed.h5`
- `preprocessed/S05_preprocessed.h5`
- All unit tests passing
- LDA NMSE + R² reported for both subjects
- HDF5 files ready to upload to Google Drive for Week 2

## GitHub
- Repo: https://github.com/Qwant07/rl-eeg-cursor
- Pushed: source code, tests, Memory notes, PDF/tex docs, preprocessed/ folder (no .h5)
- Codex CLI installed (v0.118.0, authenticated)

## Links
- [[BCI Project Index]]
- [[Architecture Decisions]]
