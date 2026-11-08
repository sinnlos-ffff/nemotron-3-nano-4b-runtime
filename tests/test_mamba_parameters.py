from dataclasses import replace

import pytest
import torch
from torch import nn

from nemotron.config import MambaConfig
from nemotron.mamba2 import Mamba2


def test_registered_checkpoint_shapes():
    mixer = Mamba2(MambaConfig.tiny())
    expected = {
        "in_proj.weight": (80, 16),
        "conv1d.weight": (44, 1, 4),
        "conv1d.bias": (44,),
        "A_log": (4,),
        "D": (4,),
        "dt_bias": (4,),
        "norm.weight": (32,),
        "out_proj.weight": (16, 32),
    }
    assert {name: tuple(value.shape) for name, value in mixer.state_dict().items()} == expected
    assert set(dict(mixer.named_parameters())) == set(expected)
    assert all(parameter.requires_grad for parameter in mixer.parameters())


def test_module_settings_and_initial_values():
    config = MambaConfig.tiny()
    mixer = Mamba2(config)
    assert isinstance(mixer.in_proj, nn.Linear)
    assert isinstance(mixer.out_proj, nn.Linear)
    assert isinstance(mixer.conv1d, nn.Conv1d)
    assert mixer.conv1d.groups == 44
    assert mixer.conv1d.kernel_size == (4,)
    assert mixer.conv1d.padding == (3,)
    assert mixer.norm.group_size == 16
    assert mixer.norm.eps == config.layer_norm_epsilon
    torch.testing.assert_close(mixer.A_log, torch.arange(1, 5, dtype=torch.float32).log())
    for parameter in (mixer.D, mixer.dt_bias, mixer.norm.weight):
        torch.testing.assert_close(parameter, torch.ones_like(parameter))


def test_bias_flags_and_module_dtype_conversion():
    config = replace(MambaConfig.tiny(), use_bias=True, use_conv_bias=False)
    mixer = Mamba2(config).to(dtype=torch.float64)
    assert mixer.in_proj.bias.shape == (80,)
    assert mixer.out_proj.bias.shape == (16,)
    assert mixer.conv1d.bias is None
    assert "conv1d.bias" not in mixer.state_dict()
    assert all(parameter.dtype == torch.float64 for parameter in mixer.parameters())
