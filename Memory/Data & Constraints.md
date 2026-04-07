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

## Secondary Dataset
- Shin et al. 2022 (Closed-loop Motor Imagery EEG Simulation)
- Located at: `project_cursor/data/20383716.zip`
- 10 subjects, 1D/2D cursor control — available for future use

## Links
- [[BCI Project Index]]
- [[Project Overview]]
