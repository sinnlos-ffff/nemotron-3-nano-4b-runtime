"""Learner-owned Mamba-2 implementation. Fill only the current exercise."""

from dataclasses import dataclass
import torch
from torch import Tensor, float32, nn

from .config import MambaConfig


@dataclass
class MambaState:
    """conv: [batch, conv_dim, kernel]; ssm: [batch, heads, head_dim, state]."""

    conv: Tensor
    ssm: Tensor


class GatedRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, group_size: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=torch.float32))
        self.group_size = group_size
        self.eps = eps

    def forward(self, hidden_states: Tensor, gate: Tensor) -> Tensor:
        """Groupwise normalization of the gated input; implemented later."""
        raise NotImplementedError("Exercise 06: grouped gated RMS normalization")


class Mamba2(nn.Module):
    def __init__(self, config: MambaConfig):
        super().__init__()
        self.config = config

        # Transformations

        self.in_proj = nn.Linear(
            in_features=config.hidden_size,
            out_features=config.projection_size,
            bias=config.use_bias
        )
        # Depthwise causal convolution
        self.conv1d = nn.Conv1d(
            in_channels=config.conv_dim,
            out_channels=config.conv_dim,
            kernel_size=config.conv_kernel,
            groups=config.conv_dim,
            padding=config.conv_kernel - 1,
            bias=config.use_conv_bias
        )
        self.out_proj = nn.Linear(
            in_features=config.inner_size,
            out_features=config.hidden_size,
            bias=config.use_bias
        )

        # Per-head parameters
        self.A_log = nn.Parameter(
            torch.log(torch.arange(1, config.mamba_num_heads + 1, dtype=torch.float32))
        )
        self.D = nn.Parameter(torch.ones(config.mamba_num_heads, dtype=torch.float32))
        self.dt_bias = nn.Parameter(torch.ones(config.mamba_num_heads, dtype=torch.float32))

        self.norm = GatedRMSNorm(
            hidden_size=config.inner_size,
            group_size=config.inner_size // config.n_groups,
            eps=config.layer_norm_epsilon
        )

    def forward(self, hidden_states: Tensor) -> Tensor:
        """[batch, length, hidden] -> same shape; initially no cache."""
        raise NotImplementedError("Exercises 03–06: implement the Mamba data flow")

    def step(self, hidden_states: Tensor, state: MambaState) -> tuple[Tensor, MambaState]:
        """[batch, 1, hidden] -> output and updated recurrent state."""
        raise NotImplementedError("Exercise 07: cached decoding")
