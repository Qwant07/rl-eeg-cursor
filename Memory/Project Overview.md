---
tags: [memory, project]
---

# Project Overview

**Goal:** Closed-loop EEG cursor control via Constrained RL (offline-to-online simulation)

## Pipeline
Raw EEG → EEGNet or LSTM decoder (frozen) → PPO agent (corrective offset) → Gymnasium sim with latency

## Core Research Question
Does **Constrained PPO** (smoothness + zeroness penalties) beat Naive PPO and static decoders?

## Conditions Compared
| Condition | Decoder | Policy |
|---|---|---|
| LDA-Offline | LDA | None |
| EEGNet-Offline | EEGNet | None |
| EEGNet-BC | EEGNet | Behavioral Cloning |
| EEGNet-NaivePPO | EEGNet | Naive PPO |
| EEGNet-ConstrainedPPO | EEGNet | Constrained PPO |
| LSTM-NaivePPO | LSTM | Naive PPO |
| LSTM-ConstrainedPPO | LSTM | Constrained PPO |

## 6-Week Sprint (Enhanced — see [[Enhanced Project Plan]] for details)
| Week | Dates | Deliverable | Where | Status |
|---|---|---|---|---|
| 1 | Apr 1–7 | Preprocessing + HDF5 + LDA baseline | Local | ✅ Done |
| 2 | Apr 8–12 | EEGNet + LSTM decoders | Colab | 🔄 Active |
| 3 | Apr 13–18 | Gymnasium env + encoder + RAP + curriculum | Colab | ⬜ |
| 4 | Apr 20–26 | Naive PPO + Constrained PPO + BC | Colab | ⬜ |
| 5 | Apr 27–May 1 | Evaluation + ablations + Shannon–Welford | Colab | ⬜ |
| 6 | May 4–8 | Report + poster + video | Local | ⬜ |

## Key Methods (from Deep Research Report)
- **Constrained PPO**: PPO + KL penalties for smoothness + zeroness
- **RAP (Real-Time Adaptive Pooling)**: adapt offline decoders for online sliding-window use
- **Neural Encoder**: synthetic EEG forward model (Shin et al. 2022 methodology)
- **Evaluation**: FTT, DIT, path efficiency, Shannon–Welford (not Fitts' Law)
- **Statistics**: repeated-measures ANOVA, post-hoc pairwise, mean ± SEM

## Links
- [[BCI Project Index]]
- [[Data & Constraints]]
- [[Architecture Decisions]]
- [[Enhanced Project Plan]]
