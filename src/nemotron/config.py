"""Configuration transport; architecture calculations are learner exercises."""

import json
from dataclasses import dataclass
from pathlib import Path


MODEL_ID = "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16"


@dataclass(frozen=True)
class MambaConfig:
    hidden_size: int
    mamba_num_heads: int
    mamba_head_dim: int
    n_groups: int
    ssm_state_size: int
    conv_kernel: int = 4
    layer_norm_epsilon: float = 1e-5
    use_bias: bool = False
    use_conv_bias: bool = True

    @classmethod
    def from_json(cls, path: str | Path) -> "MambaConfig":
        values = json.loads(Path(path).read_text())
        return cls(**{name: values[name] for name in cls.__dataclass_fields__ if name in values})

    @classmethod
    def tiny(cls) -> "MambaConfig":
        return cls(hidden_size=16, mamba_num_heads=4, mamba_head_dim=8,
                   n_groups=2, ssm_state_size=3)

    @property
    def inner_size(self) -> int:
        return self.mamba_num_heads * self.mamba_head_dim

    @property
    def conv_dim(self) -> int:
        """Exercise 01: combined width of x, B, and C."""
        return self.inner_size + (2 * self.n_groups * self.ssm_state_size)

    @property
    def projection_size(self) -> int:
        """Exercise 01: combined width of gate, x, B, C, and dt."""
        return self.inner_size + self.conv_dim + self.mamba_num_heads
