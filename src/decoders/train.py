"""Training loop for EEGNet and LSTM decoders.

Loads preprocessed HDF5 data, trains with MSE + L2 regularisation,
cosine LR decay, and early stopping. Reports NMSE and R² per axis.

Usage (local test):
    python -m src.decoders.train --subject S01 --model eegnet --epochs 50

Usage (Colab with GPU):
    python -m src.decoders.train --subject S01 --model eegnet --epochs 200 --device cuda
"""
import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from src.decoders.eegnet import EEGNet
from src.decoders.lstm import LSTMDecoder


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class EEGDataset(Dataset):
    """PyTorch dataset that loads epochs from an HDF5 file by session.

    Args:
        h5_path:    Path to the preprocessed HDF5 file.
        subject:    Subject key, e.g. "S01".
        sessions:   List of session numbers to include (1-indexed).
        max_samples: Cap total loaded epochs (0 = unlimited).
        run_types:  Which run types to load, e.g. ["AR"]. None = all.
        y_mean:     Pre-computed label mean for normalization (None = compute).
        y_std:      Pre-computed label std for normalization (None = compute).
    """

    def __init__(
        self, h5_path: str, subject: str, sessions: List[int],
        max_samples: int = 0,
        run_types: List[str] = None,
        y_mean: np.ndarray = None,
        y_std: np.ndarray = None,
    ):
        X_parts, y_parts = [], []
        total = 0
        with h5py.File(h5_path, "r") as f:
            subj_grp = f[subject]
            for sess_num in sessions:
                sess_key = f"session_{sess_num:02d}"
                if sess_key not in subj_grp:
                    continue
                sess_grp = subj_grp[sess_key]
                for run_type in sorted(sess_grp):
                    if run_types is not None and run_type not in run_types:
                        continue
                    for run_key in sorted(sess_grp[run_type]):
                        run_grp = sess_grp[run_type][run_key]
                        x = run_grp["X"][:]
                        y = run_grp["y"][:]
                        X_parts.append(x)
                        y_parts.append(y)
                        total += len(x)
                        if max_samples > 0 and total >= max_samples:
                            break
                    if max_samples > 0 and total >= max_samples:
                        break
                if max_samples > 0 and total >= max_samples:
                    break
        if not X_parts:
            raise ValueError(
                f"No data found for {subject} sessions {sessions} in {h5_path}"
            )
        self.X = np.concatenate(X_parts, axis=0).astype(np.float32)
        self.y = np.concatenate(y_parts, axis=0).astype(np.float32)
        if max_samples > 0 and len(self.X) > max_samples:
            self.X = self.X[:max_samples]
            self.y = self.y[:max_samples]

        # Z-score normalize velocity labels
        if y_mean is not None and y_std is not None:
            self.y_mean = y_mean.astype(np.float32)
            self.y_std = y_std.astype(np.float32)
        else:
            self.y_mean = self.y.mean(axis=0)
            self.y_std = self.y.std(axis=0)
            self.y_std[self.y_std == 0] = 1.0
        self.y = (self.y - self.y_mean) / self.y_std

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return torch.from_numpy(self.X[idx]), torch.from_numpy(self.y[idx])


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute NMSE and R² per velocity axis (matches BandPowerLDA.score)."""
    results = {}
    for i, axis in enumerate(["vx", "vy"]):
        mse = np.mean((y_pred[:, i] - y_true[:, i]) ** 2)
        var = np.var(y_true[:, i])
        nmse = mse / var if var > 0 else float("inf")
        ss_res = np.sum((y_true[:, i] - y_pred[:, i]) ** 2)
        ss_tot = np.sum((y_true[:, i] - np.mean(y_true[:, i])) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        results[f"nmse_{axis}"] = float(nmse)
        results[f"r2_{axis}"] = float(r2)
    results["nmse_mean"] = (results["nmse_vx"] + results["nmse_vy"]) / 2
    results["r2_mean"] = (results["r2_vx"] + results["r2_vy"]) / 2
    return results


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def build_model(name: str, n_channels: int = 62, n_times: int = 500) -> nn.Module:
    if name == "eegnet":
        return EEGNet(n_channels=n_channels, n_times=n_times)
    elif name == "lstm":
        return LSTMDecoder(n_channels=n_channels)
    else:
        raise ValueError(f"Unknown model: {name}")


@torch.no_grad()
def evaluate(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> Tuple[float, Dict[str, float]]:
    """Run model on loader, return (mean_loss, metrics_dict)."""
    model.eval()
    criterion = nn.MSELoss()
    total_loss = 0.0
    all_y, all_pred = [], []
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        pred = model(X_batch)
        total_loss += criterion(pred, y_batch).item() * len(X_batch)
        all_y.append(y_batch.cpu().numpy())
        all_pred.append(pred.cpu().numpy())
    mean_loss = total_loss / len(loader.dataset)
    y_np = np.concatenate(all_y)
    pred_np = np.concatenate(all_pred)
    metrics = compute_metrics(y_np, pred_np)
    return mean_loss, metrics


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    n_epochs: int = 200,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 20,
) -> Dict:
    """Train model with MSE + L2, cosine LR, early stopping.

    Returns dict with training history and best metrics.
    """
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    history = {
        "train_loss": [], "val_loss": [],
        "val_r2_vx": [], "val_r2_vy": [], "val_r2_mean": [],
        "val_nmse_vx": [], "val_nmse_vy": [], "val_nmse_mean": [],
        "lr": [],
    }

    best_val_loss = float("inf")
    best_state = None
    best_metrics = {}
    wait = 0

    for epoch in range(1, n_epochs + 1):
        # --- Train ---
        model.train()
        train_loss_sum = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item() * len(X_batch)
        train_loss = train_loss_sum / len(train_loader.dataset)

        # --- Validate ---
        val_loss, val_metrics = evaluate(model, val_loader, device)

        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        # --- Log ---
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["lr"].append(current_lr)
        for k in ["val_r2_vx", "val_r2_vy", "val_r2_mean",
                   "val_nmse_vx", "val_nmse_vy", "val_nmse_mean"]:
            short = k.replace("val_", "")
            history[k].append(val_metrics[short])

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"  Epoch {epoch:3d}/{n_epochs}  "
                f"train_loss={train_loss:.6f}  val_loss={val_loss:.6f}  "
                f"R²={val_metrics['r2_mean']:.4f}  "
                f"NMSE={val_metrics['nmse_mean']:.4f}  "
                f"lr={current_lr:.2e}"
            )

        # --- Early stopping ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_metrics = val_metrics
            best_metrics["epoch"] = epoch
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"  Early stopping at epoch {epoch} (patience={patience})")
                break

    # Restore best weights
    if best_state is not None:
        model.load_state_dict(best_state)

    return {"history": history, "best_metrics": best_metrics}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train EEGNet/LSTM decoder")
    parser.add_argument("--subject", type=str, required=True, help="e.g. S01")
    parser.add_argument("--model", type=str, required=True, choices=["eegnet", "lstm"])
    parser.add_argument(
        "--h5_dir", type=str, default="preprocessed",
        help="Directory containing preprocessed HDF5 files"
    )
    parser.add_argument(
        "--train_sessions", type=int, nargs="+", default=[1, 2],
        help="Session numbers for training (default: 1 2)"
    )
    parser.add_argument(
        "--val_sessions", type=int, nargs="+", default=[3],
        help="Session numbers for validation (default: 3)"
    )
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--out_dir", type=str, default="results",
        help="Directory for saving weights and history"
    )
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument(
        "--max_samples", type=int, default=0,
        help="Cap loaded epochs per split (0=unlimited). Useful for quick local tests."
    )
    parser.add_argument(
        "--run_types", type=str, nargs="+", default=["AR"],
        help="Run types to load (default: AR only). Use --run_types AR Chance for all."
    )
    args = parser.parse_args()

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    # Data
    h5_path = os.path.join(args.h5_dir, f"{args.subject}_preprocessed.h5")
    print(f"Loading training data: sessions {args.train_sessions}, run_types={args.run_types} ...")
    train_ds = EEGDataset(
        h5_path, args.subject, args.train_sessions,
        max_samples=args.max_samples, run_types=args.run_types,
    )
    print(f"  → {len(train_ds)} training epochs")
    print(f"Loading validation data: sessions {args.val_sessions} ...")
    val_ds = EEGDataset(
        h5_path, args.subject, args.val_sessions,
        max_samples=args.max_samples, run_types=args.run_types,
        y_mean=train_ds.y_mean, y_std=train_ds.y_std,
    )
    print(f"  → {len(val_ds)} validation epochs")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
    )

    # Model
    n_channels = train_ds.X.shape[1]
    n_times = train_ds.X.shape[2]
    model = build_model(args.model, n_channels=n_channels, n_times=n_times)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {args.model} ({n_params:,} parameters)")

    # Train
    t0 = time.time()
    result = train(
        model, train_loader, val_loader, device,
        n_epochs=args.epochs, lr=args.lr,
        weight_decay=args.weight_decay, patience=args.patience,
    )
    elapsed = time.time() - t0
    print(f"\nTraining finished in {elapsed:.1f}s")

    # Best metrics
    bm = result["best_metrics"]
    print(f"\nBest validation (epoch {bm['epoch']}):")
    print(f"  R²  vx={bm['r2_vx']:.4f}  vy={bm['r2_vy']:.4f}  mean={bm['r2_mean']:.4f}")
    print(f"  NMSE vx={bm['nmse_vx']:.4f} vy={bm['nmse_vy']:.4f} mean={bm['nmse_mean']:.4f}")

    # Save
    out_dir = Path(args.out_dir) / args.subject / args.model
    out_dir.mkdir(parents=True, exist_ok=True)

    weights_path = out_dir / "best_model.pt"
    torch.save(model.state_dict(), weights_path)
    print(f"Saved weights → {weights_path}")

    # Save y normalization stats for inference
    norm_path = out_dir / "y_norm.json"
    with open(norm_path, "w") as f:
        json.dump({
            "y_mean": train_ds.y_mean.tolist(),
            "y_std": train_ds.y_std.tolist(),
        }, f, indent=2)
    print(f"Saved y normalization → {norm_path}")

    history_path = out_dir / "history.json"
    with open(history_path, "w") as f:
        json.dump(result["history"], f, indent=2)
    print(f"Saved history → {history_path}")

    metrics_path = out_dir / "best_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(result["best_metrics"], f, indent=2)
    print(f"Saved metrics → {metrics_path}")


if __name__ == "__main__":
    main()
