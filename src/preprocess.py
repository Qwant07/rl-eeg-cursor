#!/usr/bin/env python3
"""Preprocess all .mat runs for a subject and write epochs to HDF5.

Usage::

    python -m src.preprocess --subject S01
    python -m src.preprocess --subject S05 --data_dir data --out_dir preprocessed
"""
import argparse
import glob
import os
import re
import sys

import numpy as np

from src.data.loader import load_run
from src.data.preprocessor import filter_raw, epoch_run, zscore_normalize
from src.data.hdf5_writer import HDF5Writer

# Regex matches filenames like: S01_Se01_AR_R01.mat
_FNAME_RE = re.compile(r'^(\w+)_Se(\d+)_(\w+)_R(\d+)\.mat$')


def _parse_filename(mat_path: str):
    """Return (subject, session, run_type, run_num) or None on mismatch."""
    basename = os.path.basename(mat_path)
    m = _FNAME_RE.match(basename)
    if not m:
        return None
    subject, session, run_type, run_num = m.groups()
    return subject, int(session), run_type, int(run_num)


def preprocess_subject(subject: str, data_dir: str, out_dir: str) -> None:
    mat_files = sorted(glob.glob(os.path.join(data_dir, subject, "*.mat")))
    if not mat_files:
        print(f"[ERROR] No .mat files found under {data_dir}/{subject}/")
        sys.exit(1)

    out_path = os.path.join(out_dir, f"{subject}_preprocessed.h5")
    print(f"Output: {out_path}")
    print(f"Found {len(mat_files)} run(s) for {subject}")

    with HDF5Writer(out_path) as writer:
        for mat_path in mat_files:
            parsed = _parse_filename(mat_path)
            if parsed is None:
                print(f"  [SKIP] Unrecognized filename: {os.path.basename(mat_path)}")
                continue

            subj, session, run_type, run_num = parsed
            tag = f"{subj} Se{session:02d} {run_type} R{run_num:02d}"
            print(f"  Processing {tag} ...", end=" ", flush=True)

            eeg, cursor_vel, cursor_pos, target_pos, eeg_times, pos_times, fs = load_run(mat_path)
            raw = filter_raw(eeg, fs)
            X, y, cursor, target = epoch_run(
                raw, eeg_times, cursor_vel, cursor_pos, target_pos, pos_times
            )
            X_norm, mean, std = zscore_normalize(X)

            writer.write_run(subj, session, run_type, run_num, X_norm, y, cursor, target, mean, std)
            print(f"{X_norm.shape[0]} epochs")

    print(f"Done → {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess EEG .mat files and write HDF5 epochs."
    )
    parser.add_argument("--subject",  required=True, help="Subject ID, e.g. S01")
    parser.add_argument("--data_dir", default="data",        help="Root data directory")
    parser.add_argument("--out_dir",  default="preprocessed", help="Output directory")
    args = parser.parse_args()

    preprocess_subject(args.subject, args.data_dir, args.out_dir)


if __name__ == "__main__":
    main()
