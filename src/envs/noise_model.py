"""Decoder noise model — simulates the EEG decoder as a noisy linear channel.

Given an intended velocity, produces what the trained decoder would predict.
Fitted from empirical decoder validation data:
    decoded_vel = gain * intended_vel + bias + N(0, noise_cov)

This replaces a full neural encoder→decoder chain with a compact statistical
model that captures the decoder's actual gain attenuation and noise profile.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class DecoderNoiseModel:
    """Linear noise model for a trained EEG velocity decoder.

    Args:
        gain: (2,) per-axis multiplicative gain [vx, vy].
        bias: (2,) per-axis additive bias.
        noise_cov: (2, 2) covariance matrix of residual noise.
        rng: numpy random generator (for reproducibility).
    """
    gain: np.ndarray = field(default_factory=lambda: np.array([1.0, 1.0]))
    bias: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0]))
    noise_cov: np.ndarray = field(
        default_factory=lambda: np.eye(2) * 0.01
    )
    rng: np.random.Generator = field(
        default_factory=lambda: np.random.default_rng()
    )

    def __call__(self, intended_vel: np.ndarray) -> np.ndarray:
        """Map intended velocity to decoded (noisy) velocity.

        Args:
            intended_vel: (2,) intended cursor velocity [vx, vy].

        Returns:
            (2,) decoded velocity with gain attenuation and noise.
        """
        noise = self.rng.multivariate_normal([0.0, 0.0], self.noise_cov)
        return self.gain * intended_vel + self.bias + noise

    @classmethod
    def from_subject(cls, subject: str, rng: Optional[np.random.Generator] = None) -> "DecoderNoiseModel":
        """Create noise model with empirical parameters for a subject.

        Parameters were fitted from EEGNet validation predictions (session 3,
        AR-only) using linear regression: pred = gain * true + bias + noise.

        Args:
            subject: "S01" or "S05".
            rng: optional random generator for reproducibility.
        """
        params = {
            "S01": {
                "gain": np.array([0.1740, 0.3035]),
                "bias": np.array([0.0957, -0.0812]),
                "noise_cov": np.array([
                    [0.12356, -0.03797],
                    [-0.03797, 0.11657],
                ]),
            },
            "S05": {
                "gain": np.array([0.4674, 0.5876]),
                "bias": np.array([-0.1112, -0.0565]),
                "noise_cov": np.array([
                    [0.05925, 0.00521],
                    [0.00521, 0.04366],
                ]),
            },
        }
        if subject not in params:
            raise ValueError(f"Unknown subject {subject}, expected one of {list(params)}")
        p = params[subject]
        return cls(
            gain=p["gain"],
            bias=p["bias"],
            noise_cov=p["noise_cov"],
            rng=rng or np.random.default_rng(),
        )

    def to_dict(self) -> dict:
        """Serialize parameters to a JSON-safe dict."""
        return {
            "gain": self.gain.tolist(),
            "bias": self.bias.tolist(),
            "noise_cov": self.noise_cov.tolist(),
        }

    @classmethod
    def from_dict(cls, d: dict, rng: Optional[np.random.Generator] = None) -> "DecoderNoiseModel":
        """Load from a dict (e.g. from JSON)."""
        return cls(
            gain=np.array(d["gain"]),
            bias=np.array(d["bias"]),
            noise_cov=np.array(d["noise_cov"]),
            rng=rng or np.random.default_rng(),
        )
