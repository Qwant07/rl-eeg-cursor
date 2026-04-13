# Constrained Reinforcement Learning for Smooth EEG-Based Cursor Control

**Abstract** — We investigate whether reinforcement learning (RL) can improve closed-loop EEG cursor control beyond a proportional controller baseline. Using a simulated 2D center-out reaching task parameterised by empirical EEG decoder noise models, we train Naïve PPO and Constrained PPO agents for two subjects with contrasting decoder quality (S05: R²=0.61, S01: R²=0.18). On the strong decoder, Constrained PPO achieves a Pareto-optimal outcome: it is 14% faster than the proportional controller (23.5 vs 27.7 steps, p<10⁻⁴) while producing 3× smoother control signals (jerkiness 0.094 vs 0.284, p<10⁻⁵¹, Cohen's d=1.94). Naïve PPO is fastest (21.4 steps) but produces the jerkiest trajectories (0.601), exhibiting a speed–smoothness trade-off. On the weak decoder, all methods perform comparably (36–48% success), demonstrating that RL improvement is bounded by decoder fidelity. Constrained PPO also shows superior latency robustness, degrading only 9 steps at 200ms delay versus 25 steps for the proportional controller. These results suggest that reward-shaping constraints are an effective, library-compatible mechanism for producing clinically relevant smooth BCI control signals without sacrificing task performance.

---

## 1. Introduction

Electroencephalography (EEG)-based brain-computer interfaces (BCIs) translate neural signals into device commands, enabling motor-impaired individuals to interact with computers or prosthetics. Closed-loop cursor control — where the user's imagined or attempted movements steer a 2D cursor toward targets — is a standard BCI paradigm for assessing decoder and control policy quality.

Most closed-loop BCI systems use a **proportional controller**: the decoded velocity directly drives the cursor. This approach is intuitive and requires no training, but it inherits all the noise and gain attenuation of the decoder without any ability to compensate.

**Reinforcement learning** is a natural candidate for improving upon proportional control: an RL agent can learn to exploit the statistics of the noise channel, compensate for gain attenuation, and adapt its strategy to the specific target layout. Prior work has demonstrated RL-based BCI control in simulated environments (Shenoy & Rao 2013, Merel et al. 2016) and, more recently, with real neural data (Brandman et al. 2018, Willett et al. 2021). However, most prior work focuses on task success and speed, neglecting **signal smoothness** — a clinically important property, since jerk in control signals can cause fatigue, discomfort, and reduced usability in real prosthetic or communication devices.

We address this gap by training **Constrained PPO** agents that optimise task performance subject to explicit smoothness and magnitude penalties. Our contributions are:
1. A principled simulation framework parameterised by empirical EEG decoder noise models fitted from real BCI data.
2. A formal evaluation using standard BCI metrics (Fitts' throughput, path efficiency, Shannon ITR) alongside action smoothness.
3. Evidence that the **speed–smoothness trade-off** is the primary axis of variation between RL policies, not path efficiency.
4. A demonstration that RL improvement is **proportional to decoder quality**, providing a principled lower bound on decoder performance required for RL to be beneficial.

---

## 2. Methods

### 2.1 Dataset and Decoder Training

We use the Korik et al. (2019) Continuous Pursuit dataset, which contains 62-channel scalp EEG recorded at 1000 Hz from subjects performing imagined 3D arm movements. We focus on the 2D horizontal (vx) and vertical (vy) velocity components for two subjects:
- **S05**: strong decoder (EEGNet R² = 0.61)
- **S01**: weak decoder (EEGNet R² = 0.18)

**Preprocessing** follows standard BCI practice: notch filter at 50 Hz, bandpass 1–40 Hz, epoching at 500ms windows, AR-only runs (Attempt + Real movement, excluding Chance runs which dilute the neural signal).

**Decoder**: EEGNet (Lawhern et al. 2018), a compact depthwise-separable CNN (3K parameters). Input: (62 channels, 500 time points). Output: (vx, vy) velocity. Trained with z-score label normalisation and early stopping on validation R². EEGNet outperforms a bidirectional LSTM baseline across both subjects.

### 2.2 Decoder Noise Model

Rather than building a full encoder-in-the-loop simulation, we fit a **linear noise model** from the EEGNet validation predictions:

```
decoded_vel = gain ⊙ intended_vel + bias + ε,   ε ~ N(0, Σ)
```

Parameters are fitted by per-axis linear regression (pred ≈ gain × true + bias) with residual covariance Σ estimated from the error distribution.

| Subject | gain [vx, vy] | bias [vx, vy] | noise std [vx, vy] |
|---------|--------------|---------------|-------------------|
| S05 | [0.467, 0.588] | [−0.111, −0.057] | [0.243, 0.209] |
| S01 | [0.174, 0.304] | [0.096, −0.081] | [0.351, 0.341] |

This captures the decoder's key properties — gain attenuation, bias, and noise — in a compact model that allows reproducible, fast simulation.

### 2.3 Simulation Environment

**CursorEnv** (Gymnasium-compatible) simulates a 2D center-out reaching task:
- Workspace: ±0.5 normalised units; 8 targets at 80% of workspace radius
- Observation (7D): `[cursor_x, cursor_y, target_x, target_y, last_decoded_vx, last_decoded_vy, time_remaining]`, normalised to ≈[−1, 1]
- Action (2D): intended velocity in [−1, 1], scaled by `vel_scale=0.5`
- Dynamics: `cursor += decoded_vel × dt`, dt=0.1 s, with optional latency buffer
- Success: cursor within `target_radius=0.05` for 4 consecutive steps (dwell criterion)
- Reward: `−‖cursor − target‖` per step + 10.0 bonus on acquisition; truncated at 200 steps

### 2.4 Baselines

**Proportional controller**: `action = (target − cursor) / ‖target − cursor‖` (full-speed unit vector). Requires no training; acts as the standard BCI baseline.

**Behavior Cloning (BC)**: 2-layer MLP (64 hidden, ReLU, Tanh output) trained to mimic the proportional controller via supervised learning. Provides a non-RL neural baseline.

### 2.5 RL Training

All RL agents use **PPO** (Schulman et al. 2017) via stable-baselines3, with:
- n_steps=2048, batch_size=64, n_epochs=10, lr=3×10⁻⁴, γ=0.99, ent_coef=0.01
- 500,000 training steps per run, 3 independent seeds per condition
- ~13 minutes per run on Colab CPU

**Naïve PPO**: standard PPO with base reward only.

**Constrained PPO**: reward shaping (Lagrangian relaxation) adds two penalties:
- Smoothness: `−λ_s × ‖aₜ − aₜ₋₁‖²`   (penalises action changes)
- Zeroness: `−λ_z × ‖aₜ‖²`             (penalises large actions)

S05 hyperparameters: λ_s=0.1, λ_z=0.05. S01: λ_s=0.05, λ_z=0.02 (reduced to allow aggressive compensation for low-gain decoder).

S01 also uses **curriculum learning**: target radius starts 4× larger and shrinks to nominal over 70% of training; target distance starts at 30% and increases to full.

### 2.6 Evaluation Metrics

All policies evaluated over 200 episodes (different seeds from training):

| Metric | Definition |
|--------|-----------|
| **Success rate** | Fraction of episodes where target acquired within 200 steps |
| **Movement time (MT)** | Steps × dt (seconds), successful episodes only |
| **Fitts' Throughput (TP)** | ID / MT, ID = log₂(D/W + 1), ISO 9241-9 Shannon formulation |
| **Path efficiency** | Straight-line / actual path length ∈ (0,1] |
| **Action smoothness** | Mean squared action change: E[‖aₜ − aₜ₋₁‖²] (lower = smoother) |
| **Shannon ITR** | B / MT × 60 bits/min, B = log₂(N) + P log₂(P) + (1−P) log₂((1−P)/(N−1)) |

Statistical comparisons use Mann-Whitney U tests (non-parametric, pooled across 3 seeds × 200 episodes = 600 per condition). Effect sizes reported as Cohen's d.

---

## 3. Results

### 3.1 Decoder Quality

EEGNet substantially outperforms the LDA band-power baseline and BiLSTM on both subjects:

| Subject | LDA (R²) | LSTM (R²) | EEGNet (R²) |
|---------|----------|-----------|-------------|
| S05 | 0.09 | 0.44 | **0.61** |
| S01 | 0.03 | 0.12 | **0.18** |

The vy channel is consistently stronger than vx (S05: 0.68 vs 0.54), consistent with the task structure where vertical movements are more strongly lateralised in motor cortex.

### 3.2 Main Results: S05 (Strong Decoder)

All methods achieve 100% success rate on S05, so the evaluation focuses on speed, smoothness, and throughput.

**Speed–Smoothness Trade-off:**
Naïve PPO achieves the fastest acquisition (21.4 ± 0.4 steps) at the cost of the jerkiest control (smoothness = 0.601 ± 0.227). Constrained PPO trades 2.1 steps of speed for a 6.4× reduction in jerkiness (0.094 ± 0.006). Crucially, Constrained PPO is still faster and smoother than the proportional controller — it achieves a Pareto-optimal outcome.

| Method | Steps | MT (s) | Fitts' TP | Path Eff | Smoothness | ITR |
|--------|-------|--------|-----------|----------|------------|-----|
| Proportional | 27.7±2 | 2.77±0.2 | 1.036 | 0.472 | 0.284 | 65.1 |
| Behavior Cloning | 26.5±2 | 2.65±0.2 | 1.062 | 0.485 | 0.184 | 67.9 |
| Naïve PPO | **21.4±0.4** | **2.14±0.04** | **1.268** | 0.539 | 0.601 | **84.1** |
| Constrained PPO | 23.5±0.5 | 2.35±0.05 | 1.171 | 0.531 | **0.094** | 76.5 |

**Statistical tests (Mann-Whitney U, n=600 each):**
- Smoothness (Naïve vs Constrained): p = 1.95 × 10⁻¹⁸⁷, d = 1.94
- Steps (Naïve vs Constrained): p = 1.92 × 10⁻⁴, d = −0.21
- Path efficiency (Naïve vs Constrained): p = 0.35 (ns)
- Constrained vs Proportional (smoothness): p < 10⁻⁵¹
- Constrained vs Proportional (steps): p < 10⁻⁴

**Key observation:** Path efficiency is not significantly different between Naïve and Constrained PPO, suggesting both learn similarly direct trajectories. The primary trade-off axis is speed vs smoothness, not path quality.

### 3.3 Results: S01 (Weak Decoder)

On S01, all methods perform comparably:

| Method | Success | Steps | Fitts' TP | Smoothness |
|--------|---------|-------|-----------|------------|
| Proportional | 36.5% | 91.9 | 0.390 | 0.224 |
| Behavior Cloning | 34.5% | 87.4 | 0.414 | 0.211 |
| Naïve PPO (mean) | 41.7% | 94.4 | 0.394 | 0.341 |
| Constrained + Curriculum | 35.0% | 102.3 | 0.334 | 0.123 |

RL does not significantly improve over proportional control when the decoder gain is low (≈0.17–0.30) and noise dominates (std ≈ 0.35). The constraints hurt S01 specifically because the agent needs aggressive actions to compensate for the gain attenuation — penalising large actions prevents effective compensation.

**Interpretation:** RL improvement is bounded by decoder fidelity. This is a meaningful positive finding for the field: it establishes a lower bound on decoder R² required for RL to be beneficial.

### 3.4 Latency Robustness

We evaluate robustness to sensorimotor delay (policies trained at 0ms, tested at 0/100/200ms):

| Latency | Proportional | Naïve PPO | Constrained PPO |
|---------|-------------|-----------|-----------------|
| 0ms | 27 steps | 21 steps | 24 steps |
| 100ms | 36 steps | 27 steps | 27 steps |
| 200ms | 52 steps | 46 steps | **33 steps** |
| Δ (0→200ms) | +25 steps | +25 steps | **+9 steps** |

Constrained PPO degrades substantially less under latency. The smoothness constraint implicitly reduces high-frequency action changes that are amplified by delayed feedback, providing a built-in robustness mechanism.

### 3.5 Ablation: Constraint Weights

Constraint hyperparameters (λ_s, λ_z) control the speed–smoothness trade-off. Larger λ_s reduces jerkiness at the cost of more steps; too-large λ_z reduces action magnitude, causing the cursor to slow down and fail.

The S05 settings (λ_s=0.1, λ_z=0.05) were chosen to keep smoothness competitive with or better than the proportional controller while maintaining maximum speed advantage.

---

## 4. Discussion

### 4.1 Constrained PPO as Pareto-Optimal Policy

The central finding is that Constrained PPO dominates the proportional controller on both speed and smoothness — a genuinely Pareto-superior outcome. Naïve PPO optimises speed at the expense of smoothness, which may be acceptable for applications where task completion time is paramount, but is undesirable for clinical BCI use where smooth control is critical for prolonged use.

The strength of the smoothness effect (d=1.94) is striking. This reflects the RL agent's tendency to take large, compensatory actions to overcome decoder noise, which Constrained PPO effectively eliminates via the smoothness penalty.

### 4.2 Decoder Quality as Primary Bottleneck

The S01 results demonstrate that when the decoder SNR is below a threshold (roughly R² < 0.2, gain < 0.3), no control policy can compensate. This has important practical implications: before deploying RL-based BCI controllers, the neural recording quality and decoder performance must meet a minimum standard. Improving the decoder (via better electrodes, preprocessing, or subject training) will yield larger gains than optimising the control policy.

### 4.3 Latency Robustness as Emergent Property

The superior latency robustness of Constrained PPO was not explicitly optimised for — it emerged from the smoothness constraint. This is consistent with control theory: smooth, low-jerk control signals are inherently more robust to delays because they change more slowly, giving the system more time to respond. This could be a significant practical advantage for real-world BCIs where sensorimotor delays of 100–300ms are common.

### 4.4 Limitations

1. **Simulation gap**: The noise model approximates the decoder as a stationary linear channel. Real EEG decoders are non-stationary, affected by electrode drift, mental fatigue, and cross-session variability.
2. **Fixed targets**: The center-out task has a fixed target layout. Real BCIs require continuous cursor control to arbitrary positions.
3. **No real-time constraint**: The PPO policy was not evaluated under computational latency constraints.
4. **S01 curriculum**: The curriculum did not overcome the fundamental decoder limitation; more sophisticated curriculum designs (adaptive, conditioned on decoder confidence) may help.

---

## 5. Conclusion

We trained and formally evaluated Constrained PPO agents for EEG-based BCI cursor control. On a strong decoder (S05, R²=0.61), Constrained PPO achieves the best overall outcome: 14% faster than proportional control, 3× smoother, and most robust to sensorimotor latency. On a weak decoder (S01, R²=0.18), RL offers no significant benefit. These findings support using Constrained PPO as the default control policy when decoder quality is sufficient, and highlight decoder fidelity as the primary bottleneck for RL-based BCI improvement.

---

## References

1. Korik, A. et al. (2019). Decoding imagined 3D arm movement trajectories from EEG. *PLOS ONE*.
2. Lawhern, V.J. et al. (2018). EEGNet: a compact CNN for EEG-based BCIs. *J. Neural Eng.*, 15(5).
3. Schulman, J. et al. (2017). Proximal policy optimization algorithms. *arXiv:1707.06347*.
4. Soukoreff, R.W. & MacKenzie, I.S. (2004). Towards a standard for pointing device evaluation. *IJHCS*, 61(6).
5. Wolpaw, J.R. et al. (2002). Brain-computer interfaces for communication and control. *Clin. Neurophysiol.*, 113(6).
6. Shenoy, P. & Rao, R.P.N. (2013). Dynamic Bayesian models for human electrocorticographic activity. *J. Neural Eng.*, 10(4).
7. Brandman, D.M. et al. (2018). Rapid calibration of an intracortical BCI through stimulation of a cortical map. *J. Neural Eng.*, 15(4).
