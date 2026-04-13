---
tags: [memory, plan, active]
---

# Enhanced Project Plan (Deep Research Report)

Based on the deep research report analysis. Updated 2026-04-12.

## Timeline Overview

| Week | Dates | Phase | Status |
|------|-------|-------|--------|
| 1 | Apr 1–7 | Preprocessing + HDF5 + LDA baseline | ✅ DONE |
| 2 | Apr 8–12 | EEGNet + LSTM decoders | ✅ DONE |
| 3 | Apr 13–18 | Gymnasium env + latency + curriculum | ✅ DONE |
| 4 | Apr 20–26 | PPO + Constrained PPO + BC | ⬜ |
| 5 | Apr 27–May 1 | Evaluation + ablations | ⬜ |
| 6 | May 4–8 | Report + poster + video | ⬜ |

---

## Week 2: Deep Decoder Development (Apr 8–12)

### Goals
- Train EEGNet and LSTM models for offline velocity decoding
- Both must beat LDA baseline (NMSE / R²)

### Tasks
- [x] Fix data path: extracted fresh from `../data.zip`, removed duplicate `data (2)/`
- [x] Run `src/preprocess.py` on S01 → `preprocessed/S01_preprocessed.h5` (6.8 GB, ~1268 epochs/run)
- [x] Run `src/preprocess.py` on S05 → `preprocessed/S05_preprocessed.h5` (15 GB, ~1268 epochs/run)
- [x] Upload HDF5 to Google Drive for Colab
- [x] Implement `src/decoders/eegnet.py`
  - Compact CNN from Lawhern et al. 2018
  - Input: `(batch, 62, 500)` → Output: `(batch, 2)` velocity
  - Temporal conv → depthwise spatial conv → separable conv → dense
- [x] Implement `src/decoders/lstm.py`
  - 2-layer bidirectional LSTM, hidden_size=128
  - Input: `(batch, 62, 500)` → Output: `(batch, 2)` velocity
- [x] Training loop (`src/decoders/train.py`)
  - `EEGDataset`: loads from HDF5 by session, supports both models
  - Loss: MSE + L2 (weight_decay in Adam), cosine LR decay
  - Early stopping on validation loss (patience=20)
  - `compute_metrics()`: NMSE + R² per axis (matches LDA scorer)
  - CLI: `python -m src.decoders.train --subject S01 --model eegnet`
  - Saves: `results/<subject>/<model>/best_model.pt`, `history.json`, `best_metrics.json`
  - 12 tests passing (`tests/decoders/test_train.py`)
- [x] Compare EEGNet vs LSTM vs LDA — report NMSE, R² per axis
  - S01: EEGNet R²=0.176, LSTM R²=0.156, LDA R²=0.052
  - S05: EEGNet R²=0.606, LSTM R²=0.471, LDA R²=0.075
- [x] Save model weights + training curves to Google Drive
- [x] Local pipeline test: EEGNet on S01 (2000 samples, 5 epochs) — R²=0.029, verified end-to-end
- Note: LSTM too slow for CPU testing (>10 min/epoch) — requires GPU

### Success Criteria
- ✅ Neural decoders exceed LDA baseline R² (EEGNet beats LDA by 3-8x)
- ✅ Training is stable (smooth loss curves, early stopping at epoch 10-54)

### Key References
- EEGNet: Lawhern et al. 2018 (compact CNN, cross-paradigm)
- LSTM for BCI: Sussillo et al. 2012 (RNN > Kalman in cursor tasks)

---

## Week 3: Simulation Environment (Apr 13–18)

### Goals
- Build Gymnasium-based closed-loop cursor simulator
- Validate end-to-end decoder → agent loop

### Tasks
- [x] Implement `src/envs/cursor_env.py` — Gymnasium environment
  - **State space**: cursor_pos (2D), target_pos (2D), last_decoded_vel (2D), time_remaining
  - **Action space**: continuous intended velocity in [-1,1]^2
  - **Reward**: -distance + success_bonus on dwell acquisition
  - **Dynamics**: cursor_pos += noise_model(action × vel_scale) × dt
  - **Realistic latency**: configurable delay buffer (0/100/200ms tested)
- [x] Implement decoder noise model (`src/envs/noise_model.py`)
  - Replaces full neural encoder with statistical model: decoded_vel = gain × intended + bias + N(0,Σ)
  - Parameters fitted from EEGNet validation data (S01 and S05)
  - More practical than full encoder→decoder chain, equally valid for RL training
- [x] Curriculum learning setup (`src/envs/curriculum.py`)
  - CurriculumWrapper: gradually shrinks target radius, increases target distance
  - S01 success rate: 46% (hard) → 92% (easy curriculum start)
- [x] Validate: proportional controller through full loop
  - No noise: 100/100, S05: 100/100, S01: 46/100
- [x] Latency ablation (proportional controller baseline)
  - S05: 100% → 99% as latency 0→200ms
  - S01: 46% → 35% as latency 0→200ms
- [x] PPO integration sanity check (stable-baselines3)
- [x] Paper figures: trajectory plots, latency bar charts
- [x] Baseline metrics saved to `results/baselines/proportional_controller.json`

### Success Criteria
- ✅ End-to-end loop works: noise_model → cursor moves → reward returned
- ✅ Simulator reproduces qualitatively reasonable cursor trajectories
- ✅ PPO integration confirmed with stable-baselines3

---

## Week 4: RL Training (Apr 20–26)

### Goals
- Train PPO agents (Naïve + Constrained) and Behavior Cloning baseline
- Constrained PPO should produce smoother trajectories

### Tasks
- [ ] Implement Behavior Cloning (BC) baseline
  - Supervised: clone optimal corrective actions from ground-truth trajectories
  - Use as initialization for PPO (optional warm-start)
- [ ] Implement Naïve PPO agent
  - Use stable-baselines3 PPO
  - State: full observation from env
  - Action: corrective velocity offset
  - Reward: shaped reward (distance + dwell bonus)
- [ ] Implement Constrained PPO
  - Modify PPO loss with KL penalties:
    - **Smoothness penalty**: penalize large action changes between steps
    - **Zeroness penalty**: push action distribution toward zero-mean Gaussian
  - Augment reward or add Lagrangian constraint terms
- [ ] Train with multiple seeds (≥3) for stability
- [ ] Curriculum: progressive target difficulty during training
- [ ] Log with TensorBoard or W&B: reward curves, action distributions, trajectory plots
- [ ] Save policy checkpoints at regular intervals

### Success Criteria
- PPO agents reach >80% target acquisition rate
- Constrained PPO yields visibly smoother trajectories vs Naïve
- Positive reward trend across training

---

## Week 5: Evaluation & Ablations (Apr 27–May 1)

### Goals
- Rigorous comparison of all conditions
- Statistical significance on key metrics

### Metrics
- **First-Touch Time (FTT)**: time to first target acquisition
- **Dial-In Time (DIT)**: time from first touch to stable dwell
- **Path Efficiency**: straight-line distance ratio, max deviation
- **Shannon–Welford Index** (instead of Fitts' Law — more robust for BCI)

### Tasks
- [ ] Evaluate all 7 conditions (see [[Project Overview]]) on held-out sessions 4–8
- [ ] Compute per-session metrics for each condition
- [ ] Ablation studies:
  - PPO with vs without smoothness penalty
  - PPO with vs without zeroness penalty
  - Different encoder delay values (100ms, 200ms, 300ms)
  - EEGNet vs LSTM as base decoder
- [ ] Statistical tests:
  - Repeated-measures ANOVA or Friedman test across sessions
  - Post-hoc pairwise comparisons (Tukey or Wilcoxon)
  - Report mean ± SEM with error bars
- [ ] Generate figures: learning curves, trajectory plots, bar charts with significance
- [ ] Per-session analysis across sessions 4–8 (replaces cross-subject evaluation)

### Success Criteria
- Clear evidence that Constrained PPO outperforms Naïve PPO (p < 0.05)
- All metrics computed and tabulated

---

## Week 6: Report & Deliverables (May 4–8)

### Paper Outline
1. **Abstract**: problem, contributions (simulator + Constrained PPO + comparisons), results
2. **Introduction**: BCI cursor control challenges, RL motivation, simulator rationale
3. **Related Work**: offline decoders (EEGNet, LSTM), simulators (Shin 2022), RL in BMI (Benton 2022)
4. **Methods**: data, encoders, decoders, RL agent (MDP + Constrained PPO), evaluation metrics
5. **Experiments**: offline decoding, closed-loop sim, RL training, ablations
6. **Results**: tables + figures + significance tests
7. **Discussion**: interpretation, limitations (sim vs real, 2 subjects only), future work
8. **Conclusion**: contributions + open science (code release)

### Deliverables
- [ ] Draft paper (IEEE/journal template)
- [ ] Poster
- [ ] Summary video
- [ ] Clean GitHub repo with README, requirements, reproduction instructions
- [ ] Reproducibility checklist (ML Reproducibility Checklist)

---

## Data Path Fix Needed

> [!warning] Data Location Issue
> The `.mat` files are in `data (2)/data/S01/` and `data (2)/data/S05/`.
> The expected path `data/data/S01/` is **empty**.
> Need to copy/symlink before running `src/preprocess.py`.

## Key Tools & Libraries
- **Preprocessing**: MNE-Python, h5py
- **Decoders**: PyTorch (EEGNet, LSTM)
- **RL**: stable-baselines3 (PPO), custom Constrained PPO wrapper
- **Simulator**: Gymnasium
- **Logging**: TensorBoard / Weights & Biases
- **Stats**: scipy.stats, statsmodels

## Links
- [[BCI Project Index]]
- [[Project Overview]]
- [[Week 1 Progress]]
- [[Data & Constraints]]
