"""Tests for the decoder training loop.

Uses a small synthetic HDF5 file so tests run in seconds without real data.
"""
import json
import os
import tempfile

import h5py
import numpy as np
import pytest
import torch

from src.decoders.train import EEGDataset, build_model, compute_metrics, evaluate, train
from torch.utils.data import DataLoader


N_CH = 62
N_TIMES = 500
N_EPOCHS_PER_RUN = 20
N_OUTPUTS = 2


@pytest.fixture
def synthetic_h5(tmp_path):
    """Create a small synthetic HDF5 file mimicking preprocessed data."""
    h5_path = str(tmp_path / "S01_preprocessed.h5")
    rng = np.random.RandomState(42)
    with h5py.File(h5_path, "w") as f:
        for sess in [1, 2, 3]:
            for run in [1, 2]:
                key = f"S01/session_{sess:02d}/AR/run_{run:02d}"
                grp = f.create_group(key)
                grp.create_dataset("X", data=rng.randn(N_EPOCHS_PER_RUN, N_CH, N_TIMES).astype(np.float32))
                grp.create_dataset("y", data=rng.randn(N_EPOCHS_PER_RUN, N_OUTPUTS).astype(np.float32))
                grp.create_dataset("cursor", data=rng.randn(N_EPOCHS_PER_RUN, 2).astype(np.float32))
                grp.create_dataset("target", data=rng.randn(N_EPOCHS_PER_RUN, 2).astype(np.float32))
                grp.create_dataset("mean", data=rng.randn(N_CH).astype(np.float64))
                grp.create_dataset("std", data=np.abs(rng.randn(N_CH)).astype(np.float64) + 0.1)
            # Add a Chance run per session
            key = f"S01/session_{sess:02d}/Chance/run_01"
            grp = f.create_group(key)
            grp.create_dataset("X", data=rng.randn(N_EPOCHS_PER_RUN, N_CH, N_TIMES).astype(np.float32))
            grp.create_dataset("y", data=rng.randn(N_EPOCHS_PER_RUN, N_OUTPUTS).astype(np.float32))
            grp.create_dataset("cursor", data=rng.randn(N_EPOCHS_PER_RUN, 2).astype(np.float32))
            grp.create_dataset("target", data=rng.randn(N_EPOCHS_PER_RUN, 2).astype(np.float32))
            grp.create_dataset("mean", data=rng.randn(N_CH).astype(np.float64))
            grp.create_dataset("std", data=np.abs(rng.randn(N_CH)).astype(np.float64) + 0.1)
    return h5_path


class TestEEGDataset:
    def test_loads_correct_sessions(self, synthetic_h5):
        ds = EEGDataset(synthetic_h5, "S01", sessions=[1, 2], run_types=["AR"])
        # 2 sessions × 2 AR runs × 20 epochs = 80
        assert len(ds) == 80

    def test_single_session(self, synthetic_h5):
        ds = EEGDataset(synthetic_h5, "S01", sessions=[3], run_types=["AR"])
        assert len(ds) == 40  # 1 session × 2 AR runs × 20

    def test_shapes(self, synthetic_h5):
        ds = EEGDataset(synthetic_h5, "S01", sessions=[1], run_types=["AR"])
        X, y = ds[0]
        assert X.shape == (N_CH, N_TIMES)
        assert y.shape == (N_OUTPUTS,)
        assert X.dtype == torch.float32

    def test_missing_session_raises(self, synthetic_h5):
        with pytest.raises(ValueError):
            EEGDataset(synthetic_h5, "S01", sessions=[99], run_types=["AR"])

    def test_run_types_filter(self, synthetic_h5):
        """AR-only should exclude Chance runs."""
        ds_ar = EEGDataset(synthetic_h5, "S01", sessions=[1], run_types=["AR"])
        ds_all = EEGDataset(synthetic_h5, "S01", sessions=[1], run_types=None)
        # AR: 2 runs × 20 = 40; All: 2 AR + 1 Chance = 60
        assert len(ds_ar) == 40
        assert len(ds_all) == 60

    def test_y_normalization(self, synthetic_h5):
        """Y should be z-score normalized."""
        ds = EEGDataset(synthetic_h5, "S01", sessions=[1], run_types=["AR"])
        # Normalized y should have mean ≈ 0 and std ≈ 1
        assert abs(ds.y[:, 0].mean()) < 0.2
        assert abs(ds.y[:, 1].mean()) < 0.2

    def test_val_uses_train_y_stats(self, synthetic_h5):
        """Validation set should use training set's y normalization stats."""
        train_ds = EEGDataset(synthetic_h5, "S01", sessions=[1], run_types=["AR"])
        val_ds = EEGDataset(
            synthetic_h5, "S01", sessions=[3], run_types=["AR"],
            y_mean=train_ds.y_mean, y_std=train_ds.y_std,
        )
        np.testing.assert_array_equal(val_ds.y_mean, train_ds.y_mean)
        np.testing.assert_array_equal(val_ds.y_std, train_ds.y_std)


class TestComputeMetrics:
    def test_perfect_prediction(self):
        y = np.array([[1.0, 2.0], [3.0, 4.0]])
        m = compute_metrics(y, y)
        assert m["nmse_mean"] == pytest.approx(0.0, abs=1e-10)
        assert m["r2_mean"] == pytest.approx(1.0, abs=1e-10)

    def test_has_all_keys(self):
        y = np.random.randn(50, 2)
        p = np.random.randn(50, 2)
        m = compute_metrics(y, p)
        for k in ["nmse_vx", "nmse_vy", "nmse_mean", "r2_vx", "r2_vy", "r2_mean"]:
            assert k in m


class TestBuildModel:
    def test_eegnet(self):
        m = build_model("eegnet")
        assert isinstance(m, torch.nn.Module)
        out = m(torch.randn(2, N_CH, N_TIMES))
        assert out.shape == (2, N_OUTPUTS)

    def test_lstm(self):
        m = build_model("lstm")
        out = m(torch.randn(2, N_CH, N_TIMES))
        assert out.shape == (2, N_OUTPUTS)

    def test_unknown_raises(self):
        with pytest.raises(ValueError):
            build_model("unknown")


class TestTrainLoop:
    def test_short_training_runs(self, synthetic_h5):
        """Run 5 epochs and verify outputs are sane."""
        train_ds = EEGDataset(synthetic_h5, "S01", sessions=[1, 2], run_types=["AR"])
        val_ds = EEGDataset(synthetic_h5, "S01", sessions=[3], run_types=["AR"],
                            y_mean=train_ds.y_mean, y_std=train_ds.y_std)
        train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=16)

        model = build_model("eegnet")
        device = torch.device("cpu")
        result = train(
            model, train_loader, val_loader, device,
            n_epochs=5, lr=1e-3, weight_decay=1e-4, patience=50,
        )

        # Check history structure
        h = result["history"]
        assert len(h["train_loss"]) == 5
        assert len(h["val_loss"]) == 5
        assert all(v > 0 for v in h["train_loss"])

        # Check best metrics exist
        bm = result["best_metrics"]
        assert "r2_mean" in bm
        assert "nmse_mean" in bm
        assert "epoch" in bm

    def test_early_stopping(self, synthetic_h5):
        """With patience=1 and random data, should stop early."""
        train_ds = EEGDataset(synthetic_h5, "S01", sessions=[1], run_types=["AR"])
        val_ds = EEGDataset(synthetic_h5, "S01", sessions=[3], run_types=["AR"],
                            y_mean=train_ds.y_mean, y_std=train_ds.y_std)
        train_loader = DataLoader(train_ds, batch_size=40)
        val_loader = DataLoader(val_ds, batch_size=40)

        model = build_model("lstm")
        device = torch.device("cpu")
        result = train(
            model, train_loader, val_loader, device,
            n_epochs=100, lr=1e-3, patience=3,
        )
        # Should have stopped well before 100 epochs
        actual_epochs = len(result["history"]["train_loss"])
        assert actual_epochs < 100

    def test_evaluate(self, synthetic_h5):
        val_ds = EEGDataset(synthetic_h5, "S01", sessions=[3], run_types=["AR"])
        val_loader = DataLoader(val_ds, batch_size=16)
        model = build_model("eegnet")
        device = torch.device("cpu")
        model.to(device)
        loss, metrics = evaluate(model, val_loader, device)
        assert loss > 0
        assert "r2_mean" in metrics
