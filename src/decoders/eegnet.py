"""EEGNet decoder for continuous velocity regression.

Architecture based on Lawhern et al. 2018 (J. Neural Eng.), adapted from
classification to regression (MSE loss, 2D velocity output).

Input:  (batch, 1, n_channels, n_times) = (batch, 1, 62, 500)
Output: (batch, 2) — predicted cursor velocity (vx, vy)
"""
import torch
import torch.nn as nn


class EEGNet(nn.Module):
    """Compact CNN for EEG decoding (Lawhern et al. 2018).

    Args:
        n_channels: number of EEG channels (default 62)
        n_times:    number of time samples per epoch (default 500)
        n_outputs:  output dimensionality (default 2 for vx, vy)
        F1:         number of temporal filters (default 8)
        D:          depth multiplier for depthwise conv (default 2)
        F2:         number of separable filters (default 16, typically F1*D)
        dropout:    dropout rate (default 0.25)
    """

    def __init__(
        self,
        n_channels: int = 62,
        n_times: int = 500,
        n_outputs: int = 2,
        F1: int = 8,
        D: int = 2,
        F2: int = 16,
        dropout: float = 0.25,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.n_times = n_times

        # Block 1: Temporal convolution
        # Kernel size = half the sampling rate for ~125ms temporal window
        self.conv1 = nn.Conv2d(1, F1, (1, 125), padding=(0, 62), bias=False)
        self.bn1 = nn.BatchNorm2d(F1)

        # Block 1: Depthwise spatial convolution (learns spatial filters)
        self.depthwise = nn.Conv2d(
            F1, F1 * D, (n_channels, 1), groups=F1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(F1 * D)
        self.act1 = nn.ELU()
        self.pool1 = nn.AvgPool2d((1, 4))  # downsample time by 4
        self.drop1 = nn.Dropout(dropout)

        # Block 2: Separable convolution
        self.separable1 = nn.Conv2d(
            F1 * D, F2, (1, 16), padding=(0, 8), groups=F1 * D, bias=False
        )
        self.separable2 = nn.Conv2d(F2, F2, (1, 1), bias=False)
        self.bn3 = nn.BatchNorm2d(F2)
        self.act2 = nn.ELU()
        self.pool2 = nn.AvgPool2d((1, 8))  # downsample time by 8
        self.drop2 = nn.Dropout(dropout)

        # Compute flattened feature size
        self._flat_size = self._get_flat_size()

        # Regression head
        self.fc = nn.Linear(self._flat_size, n_outputs)

    def _get_flat_size(self) -> int:
        """Forward a dummy tensor to compute the flattened feature size."""
        with torch.no_grad():
            x = torch.zeros(1, 1, self.n_channels, self.n_times)
            x = self._feature_forward(x)
            return x.shape[1]

    def _feature_forward(self, x: torch.Tensor) -> torch.Tensor:
        # Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.depthwise(x)
        x = self.bn2(x)
        x = self.act1(x)
        x = self.pool1(x)
        x = self.drop1(x)

        # Block 2
        x = self.separable1(x)
        x = self.separable2(x)
        x = self.bn3(x)
        x = self.act2(x)
        x = self.pool2(x)
        x = self.drop2(x)

        return x.flatten(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (batch, 1, n_channels, n_times) or (batch, n_channels, n_times)

        Returns:
            (batch, n_outputs) predicted velocity
        """
        if x.dim() == 3:
            x = x.unsqueeze(1)  # add channel dim: (B, 1, C, T)
        x = self._feature_forward(x)
        return self.fc(x)
