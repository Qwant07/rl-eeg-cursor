"""Tests for HDF5Writer."""
import os
import tempfile
import numpy as np
import h5py
import pytest
from src.data.hdf5_writer import HDF5Writer

N, N_CH, N_T = 50, 62, 500


def _make_run_data(seed=0):
    rng = np.random.default_rng(seed)
    X      = rng.standard_normal((N, N_CH, N_T))
    y      = rng.standard_normal((N, 2))
    cursor = rng.standard_normal((N, 2))
    target = rng.standard_normal((N, 2))
    mean   = rng.standard_normal((N_CH,))
    std    = np.abs(rng.standard_normal((N_CH,))) + 1e-6
    return X, y, cursor, target, mean, std


def test_hdf5_writer_creates_file():
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "test.h5")
        X, y, cursor, target, mean, std = _make_run_data()
        with HDF5Writer(out) as writer:
            writer.write_run("S01", 1, "AR", 1, X, y, cursor, target, mean, std)
        assert os.path.exists(out)


def test_hdf5_writer_dataset_shapes():
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "test.h5")
        X, y, cursor, target, mean, std = _make_run_data()
        with HDF5Writer(out) as writer:
            writer.write_run("S01", 1, "AR", 1, X, y, cursor, target, mean, std)
        with h5py.File(out, "r") as f:
            grp = f["S01/session_01/AR/run_01"]
            assert grp["X"].shape      == (N, N_CH, N_T)
            assert grp["y"].shape      == (N, 2)
            assert grp["cursor"].shape == (N, 2)
            assert grp["target"].shape == (N, 2)
            assert grp["mean"].shape   == (N_CH,)
            assert grp["std"].shape    == (N_CH,)


def test_hdf5_writer_multiple_runs():
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "test.h5")
        with HDF5Writer(out) as writer:
            for run_num in range(1, 4):
                X, y, cursor, target, mean, std = _make_run_data(seed=run_num)
                writer.write_run("S01", 1, "AR", run_num, X, y, cursor, target, mean, std)
        with h5py.File(out, "r") as f:
            assert "S01/session_01/AR/run_01" in f
            assert "S01/session_01/AR/run_02" in f
            assert "S01/session_01/AR/run_03" in f


def test_hdf5_writer_overwrites_existing_run():
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "test.h5")
        X1, y1, c1, t1, m1, s1 = _make_run_data(seed=1)
        X2, y2, c2, t2, m2, s2 = _make_run_data(seed=2)
        with HDF5Writer(out) as writer:
            writer.write_run("S01", 1, "AR", 1, X1, y1, c1, t1, m1, s1)
        with HDF5Writer(out) as writer:
            writer.write_run("S01", 1, "AR", 1, X2, y2, c2, t2, m2, s2)
        with h5py.File(out, "r") as f:
            stored_y = f["S01/session_01/AR/run_01/y"][:]
            np.testing.assert_allclose(stored_y, y2.astype(np.float32), rtol=1e-5)


def test_hdf5_writer_values_roundtrip():
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "test.h5")
        X, y, cursor, target, mean, std = _make_run_data()
        with HDF5Writer(out) as writer:
            writer.write_run("S01", 2, "EG", 3, X, y, cursor, target, mean, std)
        with h5py.File(out, "r") as f:
            grp = f["S01/session_02/EG/run_03"]
            np.testing.assert_allclose(grp["mean"][:], mean, rtol=1e-10)
            np.testing.assert_allclose(grp["std"][:],  std,  rtol=1e-10)
            # X/y stored as float32 — check within float32 tolerance
            np.testing.assert_allclose(grp["X"][:], X.astype(np.float32), rtol=1e-5)
