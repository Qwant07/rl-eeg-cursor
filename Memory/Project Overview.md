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

## 6-Week Sprint
| Week | Deliverable | Where |
|---|---|---|
| 1 | Preprocessing + HDF5 + LDA baseline | Local |
| 2 | EEGNet + LSTM decoders | Colab |
| 3 | Gymnasium env + latency + curriculum | Colab |
| 4 | Naive PPO + Constrained PPO + BC | Colab |
| 5 | Evaluation + per-session ablations | Colab |
| 6 | Report + poster + video | Local |

## Links
- [[BCI Project Index]]
- [[Data & Constraints]]
- [[Architecture Decisions]]
