"""Learner-owned Mamba-2 implementation. Fill only the current exercise."""

from dataclasses import dataclass

from torch import Tensor, nn

from .config import MambaConfig


@dataclass
class MambaState:
    """conv: [batch, conv_dim, kernel]; ssm: [batch, heads, head_dim, state]."""

    conv: Tensor
    ssm: Tensor


class Mamba2(nn.Module):
    def __init__(self, config: MambaConfig):
        super().__init__()
        self.config = config
        raise NotImplementedError("Exercise 02: register projections and parameters")

    def forward(self, hidden_states: Tensor) -> Tensor:
        """[batch, length, hidden] -> same shape; initially no cache."""
        raise NotImplementedError("Exercises 03–06: implement the Mamba data flow")

    def step(self, hidden_states: Tensor, state: MambaState) -> tuple[Tensor, MambaState]:
        """[batch, 1, hidden] -> output and updated recurrent state."""
        raise NotImplementedError("Exercise 07: cached decoding")
