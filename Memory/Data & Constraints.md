---
tags: [memory, data]
---

# Data & Constraints

## Dataset
- **Primary:** Continuous Pursuit EEG BCI Dataset (Korik et al., 2024)
- **Subjects:** S01 and S05 only (storage constraint — 20 GB total, ~12.5 GB used by data)
- **Sessions per subject:** 8
- **Files per subject:** 104 `.mat` files
- **Run types:** AR (Adaptive Runs), EG, PN, Chance

## Actual Data Format (verified by inspection)
- `.mat` files are **HDF5 v7.3** — use `h5py`, NOT `scipy.io.loadmat`
- **62 EEG channels** (FP1…CB2)
- **1000 Hz** sampling rate
- EEG shape per run: `(317280, 62)` (~317 seconds)
- Cursor/target data: **25 Hz**, shape `(7932,)`
- `eeg/times` and `eeg/postimes` are **sample indices**, not seconds
- After epoching: window shape `(N, 62, 500)` — 500 ms @ 1000 Hz, 250 ms stride

> ⚠️ The original proposal said 512 Hz / 64 channels — both are WRONG. Real data is 1000 Hz / 62 channels.

## Session Splits
- Sessions 1–2 → decoder training
- Session 3 → RL fine-tuning
- Sessions 4–8 → held-out evaluation

## Scope Adjustment
- No cross-subject generalization (only 2 subjects)
- Per-session ablations across sessions 4–8 replace cross-subject evaluation

## Data Path Status (2026-04-08)

> [!success] Path Fixed
> Fresh extraction from `../data.zip` into `data/`. Duplicate `data (2)/` removed.
> `.mat` files now at `data/S01/` (104 files) and `data/S05/` (104 files).

- HDF5 preprocessed files generated (2026-04-08):
  - `preprocessed/S01_preprocessed.h5` — 6.8 GB, 104 runs, ~1268 epochs/run
  - `preprocessed/S05_preprocessed.h5` — 15 GB, 104 runs, ~1268 epochs/run

## Secondary Dataset
- Shin et al. 2022 (Closed-loop Motor Imagery EEG Simulation)
- Located at: `data/data/20383716.zip` and `data (2)/data/20383716.zip`
- 10 subjects, 1D/2D cursor control — needed for Week 3 (simulator validation)

## Additional Datasets (from Deep Research Report)
- PhysioNet EEG MI (109 subjects, cued MI) — optional pretraining
- BCI Competition IV 2a (9 subjects, 4-class MI) — benchmark
- Forenzo et al. 2024 is the primary dataset we already have (S01, S05)

## Links
- [[BCI Project Index]]
- [[Project Overview]]
- [[Enhanced Project Plan]]
