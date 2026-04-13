# Constrained RL for Closed-Loop EEG Cursor Control

> **Course project — BCI (Spring 2026)**
> Reinforcement learning policies for EEG-based 2D cursor reaching, with formal evaluation using Fitts' throughput, path efficiency, and action smoothness metrics.

---

## Overview

We train and evaluate control policies for a simulated closed-loop BCI cursor task. An EEGNet decoder translates scalp EEG signals into 2D cursor velocity; RL agents learn to compensate for the decoder's gain attenuation, noise, and latency.

**Key finding:** Constrained PPO achieves a Pareto-optimal outcome — it is simultaneously **14% faster** than a proportional controller (24 vs 28 steps) and **3× smoother** (jerkiness 0.094 vs 0.284), with the smoothness difference being statistically massive (Mann-Whitney p < 10⁻¹⁸⁷, Cohen's d = 1.94).

---

## Results (S05, R²=0.61 decoder)

| Method | Success | Steps (±95% CI) | Fitts' TP | Smoothness | Shannon ITR |
|--------|---------|-----------------|-----------|------------|-------------|
| Proportional Controller | 100% | 28 ± 2 | 1.036 bits/s | 0.284 | 65.1 bits/min |
| Behavior Cloning | 100% | 27 ± 2 | 1.062 bits/s | 0.184 | 67.9 bits/min |
| Naïve PPO (3 seeds) | 100% | 21.4 ± 0.4 | 1.268 bits/s | 0.601 | 84.1 bits/min |
| **Constrained PPO (3 seeds)** | **100%** | **23.5 ± 0.5** | **1.171 bits/s** | **0.094** | **76.5 bits/min** |

*200 episodes per condition. PPO metrics are mean ± std across 3 seeds.*

**Statistical significance (Mann-Whitney U, n=600 per condition):**
- Smoothness: p = 1.95 × 10⁻¹⁸⁷, Cohen's d = 1.94 (huge effect)
- Speed (steps): p = 1.92 × 10⁻⁴
- Path efficiency: p = 0.35 (ns — both policies equally efficient)

**On weak decoder (S01, R²=0.18):** RL ≈ proportional controller (36–48% vs 37%). Decoder quality is the bottleneck, not the policy. This is a valid finding: RL improvement is bounded by decoder fidelity.

**Latency robustness:** Constrained PPO degrades least under sensorimotor delay (only +9 steps at 200ms, vs +25 for proportional controller).

---

## Repository Structure

```
rl-eeg-cursor/
├── src/
│   ├── data/           # Preprocessing pipeline (MNE, HDF5)
│   ├── decoders/       # EEGNet, LSTM decoders + training
│   ├── baselines/      # LDA band-power decoder
│   ├── envs/           # Gymnasium cursor environment
│   │   ├── cursor_env.py        # 2D reaching task with noise model
│   │   ├── noise_model.py       # EEG decoder as noisy linear channel
│   │   └── curriculum.py        # Progressive difficulty scheduling
│   ├── agents/         # RL training
│   │   ├── constrained_wrapper.py  # Smoothness + zeroness penalties
│   │   ├── behavior_cloning.py     # BC baseline
│   │   └── train_ppo.py            # Unified PPO training CLI
│   └── evaluation/     # Formal BCI metrics
│       ├── metrics.py             # Fitts' TP, path efficiency, ITR
│       └── run_eval.py            # Evaluation CLI + statistical tests
├── tests/              # 106 tests (pytest)
├── requirements.txt
└── README.md
```

---

## Installation

```bash
git clone https://github.com/Qwant07/rl-eeg-cursor.git
cd rl-eeg-cursor
pip install -r requirements.txt
```

**Requirements:** Python 3.10+, PyTorch 2.3+, stable-baselines3 2.3+, gymnasium 1.0+, MNE 1.7+

---

## Reproducing Results

All experiments were run on Google Colab (T4 GPU for decoder training, CPU for RL).

### 1. Preprocess EEG data

```bash
python -m src.data.preprocessor --subject S05 --out_dir preprocessed/
```

### 2. Train EEGNet decoder

```bash
python -m src.decoders.train \
    --subject S05 --model eegnet --device cuda \
    --epochs 200 --batch_size 128 --patience 20 \
    --train_sessions 1 --out_dir results
```

### 3. Train RL policies

```bash
# Naïve PPO (3 seeds)
for seed in 0 1 2; do
    python -m src.agents.train_ppo \
        --subject S05 --mode naive --seed $seed \
        --total_steps 500000 --out_dir results
done

# Constrained PPO (3 seeds)
for seed in 0 1 2; do
    python -m src.agents.train_ppo \
        --subject S05 --mode constrained --seed $seed \
        --lambda_smooth 0.1 --lambda_zero 0.05 \
        --total_steps 500000 --out_dir results
done
```

### 4. Formal evaluation

```bash
# Evaluate a saved policy
python -m src.evaluation.run_eval eval \
    --policy ppo --model_path results/S05/naive_seed0/ppo_model.zip \
    --subject S05 --n_episodes 200 --out results/S05/naive_seed0_eval.json

# Compare two conditions
python -m src.evaluation.run_eval compare \
    results/S05/naive_seed0_eval.json \
    results/S05/constrained_seed0_eval.json
```

### 5. Run tests

```bash
pytest tests/ -v
# 106 tests, all passing
```

---

## Environment Details

**CursorEnv** (`src/envs/cursor_env.py`):
- 2D workspace ±0.5, 8 center-out targets at 80% radius
- 7D observation: `[cursor_x, cursor_y, target_x, target_y, decoded_vx, decoded_vy, time_remaining]`
- 2D continuous action: intended velocity in `[-1, 1]`
- Reward: `−distance_to_target` per step + `10.0` bonus on acquisition (4-step dwell)
- Configurable: latency buffer, curriculum, noise model

**DecoderNoiseModel** (`src/envs/noise_model.py`):
- Fitted from EEGNet validation predictions: `decoded = gain × intended + bias + N(0, Σ)`
- S05: gain ≈ [0.47, 0.59], noise std ≈ 0.24 (SNR ≈ 2:1)
- S01: gain ≈ [0.17, 0.30], noise std ≈ 0.35 (SNR < 1:1)

**Constrained PPO** (`src/agents/constrained_wrapper.py`):
- Lagrangian relaxation via reward shaping
- Smoothness penalty: `−λ_s × ‖aₜ − aₜ₋₁‖²`
- Zeroness penalty: `−λ_z × ‖aₜ‖²`
- S05: λ_s=0.1, λ_z=0.05 · S01: λ_s=0.05, λ_z=0.02

---

## Evaluation Metrics

| Metric | Formula | Notes |
|--------|---------|-------|
| Fitts' Throughput | `ID / MT`, ID = log₂(D/W + 1) | ISO 9241-9 Shannon formulation |
| Path Efficiency | `straight_dist / actual_path_len` | 1.0 = perfectly straight |
| Action Smoothness | `mean ‖aₜ − aₜ₋₁‖²` | Lower = smoother |
| Shannon ITR | `B / MT × 60` bits/min | Wolpaw et al. 2002 |

---

## Citation

If you use this code, please cite the dataset:

> Korik et al. (2019). *Decoding imagined 3D arm movement trajectories from EEG to control a robotic arm.* PLOS ONE.
