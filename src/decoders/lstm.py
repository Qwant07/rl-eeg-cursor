"""LSTM decoder for continuous velocity regression.

Takes windowed EEG epochs and predicts cursor velocity using a
bidirectional LSTM over the time dimension.

Input:  (batch, n_channels, n_times) = (batch, 62, 500)
Output: (batch, 2) — predicted cursor velocity (vx, vy)
"""
import torch
import torch.nn as nn


class LSTMDecoder(nn.Module):
    """Bidirectional LSTM decoder for EEG velocity regression.

    Args:
        n_channels:  number of EEG channels (default 62)
        hidden_size: LSTM hidden units per direction (default 128)
        n_layers:    number of stacked LSTM layers (default 2)
        n_outputs:   output dimensionality (default 2 for vx, vy)
        dropout:     dropout between LSTM layers (default 0.3)
    """

    def __init__(
        self,
        n_channels: int = 62,
        hidden_size: int = 128,
        n_layers: int = 2,
        n_outputs: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_channels,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.drop = nn.Dropout(dropout)
        # bidirectional → 2 * hidden_size
        self.fc = nn.Linear(hidden_size * 2, n_outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (batch, n_channels, n_times) — EEG epoch

        Returns:
            (batch, n_outputs) predicted velocity
        """
        # LSTM expects (batch, seq_len, features) → transpose to (B, T, C)
        x = x.transpose(1, 2)  # (B, 500, 62)
        out, _ = self.lstm(x)   # (B, 500, hidden*2)
        # Use last time step output
        last = out[:, -1, :]    # (B, hidden*2)
        last = self.drop(last)
        return self.fc(last)
