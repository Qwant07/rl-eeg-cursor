---
tags: [memory, architecture]
---

# Architecture Decisions

## Pipeline Choice: Option A (Modular scripts + HDF5)
- Preprocess locally → save HDF5 → upload to Google Drive → train on Colab
- One HDF5 file per subject (~1–1.5 GB compressed)
- Process one `.mat` run at a time (memory-safe for 8 GB RAM)

## Workflow Split
- **Local (M1 Mac):** `preprocess.py` only
- **Colab (GPU):** decoder training, RL training, evaluation

## File Structure (Expanded for Full Pipeline)
```
rl-eeg-cursor/
  src/
    data/
      loader.py          ← load_run(): h5py → numpy
      preprocessor.py    ← filter_raw(), epoch_run(), zscore_normalize()
      hdf5_writer.py     ← HDF5Writer context manager
    baselines/
      lda_decoder.py     ← BandPowerLDA
    decoders/             ← Week 2
      eegnet.py          ← EEGNet CNN decoder
      lstm.py            ← LSTM decoder
      train.py           ← shared training loop
    envs/                 ← Week 3
      cursor_env.py      ← Gymnasium cursor environment
      encoder.py         ← neural encoder (synthetic EEG)
    agents/               ← Week 4
      ppo.py             ← Naïve PPO wrapper (stable-baselines3)
      constrained_ppo.py ← Constrained PPO (KL penalties)
      bc.py              ← Behavior Cloning baseline
    evaluation/           ← Week 5
      metrics.py         ← FTT, DIT, path efficiency, Shannon–Welford
      ablations.py       ← ablation experiment runner
    preprocess.py        ← CLI script
  tests/
    data/
    baselines/
    decoders/
    envs/
    agents/
  preprocessed/          ← HDF5 output files
  Memory/                ← Obsidian vault (project notes)
```

## Key Interface
```
load_run(path) → (eeg, cursor_vel, cursor_pos, target_pos, eeg_times, pos_times, fs)
filter_raw(eeg, fs) → mne.io.RawArray
epoch_run(raw, ...) → (X, y, cursor, target)  # X: (N, 62, 500)
zscore_normalize(X) → (X_norm, mean, std)
HDF5Writer.write_run(session, run_type, run_num, X, y, cursor, target, mean, std)
```

## ICA Decision
ICA skipped — no EOG channels for automated detection, too slow per-run. Bandpass + CAR is sufficient for course project.

## Links
- [[BCI Project Index]]
- [[Project Overview]]
- [[Week 1 Progress]]
