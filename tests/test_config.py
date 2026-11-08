import json

import pytest

from nemotron.config import MambaConfig


def test_config_reads_checkpoint_fields_without_remote_code(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({
        "hidden_size": 16, "mamba_num_heads": 4, "mamba_head_dim": 8,
        "n_groups": 2, "ssm_state_size": 3, "unrelated_field": True,
    }))
    assert MambaConfig.from_json(path) == MambaConfig.tiny()


def test_tiny_dimensions():
    config = MambaConfig.tiny()
    assert config.inner_size == 32
    assert config.conv_dim == 44
    assert config.projection_size == 80


def test_checkpoint_dimensions():
    config = MambaConfig(hidden_size=3136, mamba_num_heads=96, mamba_head_dim=80,
                         n_groups=8, ssm_state_size=128)
    assert config.inner_size == 7680
    assert config.conv_dim == 9728
    assert config.projection_size == 17504
