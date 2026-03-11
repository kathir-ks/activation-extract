"""Tests for configs and presets."""

import pytest

from sae.configs.base import SAEConfig
from sae.configs.training import TrainingConfig
from sae.configs.presets import PRESETS, qwen_0_5b_vanilla, qwen_0_5b_topk


class TestSAEConfig:
    def test_defaults(self):
        cfg = SAEConfig()
        assert cfg.hidden_dim == 896
        assert cfg.dict_size == 896 * 16
        assert cfg.architecture == "vanilla"
        assert cfg.dtype == "bfloat16"
        assert cfg.normalize_decoder is True

    def test_custom(self):
        cfg = SAEConfig(hidden_dim=512, dict_size=8192, architecture="topk", k=32)
        assert cfg.hidden_dim == 512
        assert cfg.dict_size == 8192
        assert cfg.k == 32

    def test_gated_bandwidth_separate_from_jumprelu(self):
        """gated_bandwidth should be independent of jumprelu_bandwidth."""
        cfg = SAEConfig(jumprelu_bandwidth=0.01, gated_bandwidth=0.05)
        assert cfg.jumprelu_bandwidth == 0.01
        assert cfg.gated_bandwidth == 0.05


class TestTrainingConfig:
    def test_defaults(self):
        cfg = TrainingConfig()
        assert cfg.batch_size == 4096
        assert cfg.num_steps == 100_000
        assert cfg.optimizer == "adam"
        assert cfg.lr_decay == "cosine"
        assert cfg.shuffle_buffer_size == 262144

    def test_source_kwargs_not_shared(self):
        """source_kwargs should use default_factory, not be shared."""
        cfg1 = TrainingConfig()
        cfg2 = TrainingConfig()
        cfg1.source_kwargs["test"] = True
        assert "test" not in cfg2.source_kwargs


class TestPresets:
    def test_qwen_vanilla_preset(self):
        sae_cfg, train_cfg = qwen_0_5b_vanilla()
        assert sae_cfg.hidden_dim == 896
        assert sae_cfg.architecture == "vanilla"
        assert sae_cfg.dict_size == 896 * 16
        assert train_cfg.batch_size == 4096

    def test_qwen_topk_preset(self):
        sae_cfg, train_cfg = qwen_0_5b_topk(k=32)
        assert sae_cfg.architecture == "topk"
        assert sae_cfg.k == 32

    def test_preset_layer_override(self):
        sae_cfg, train_cfg = qwen_0_5b_vanilla(layer=20)
        assert train_cfg.layer_index == 20

    def test_preset_expansion_override(self):
        sae_cfg, _ = qwen_0_5b_vanilla(expansion=32)
        assert sae_cfg.dict_size == 896 * 32

    def test_all_presets_exist(self):
        assert "qwen-0.5b-vanilla" in PRESETS
        assert "qwen-0.5b-topk" in PRESETS

    def test_all_presets_callable(self):
        for name, factory in PRESETS.items():
            sae_cfg, train_cfg = factory()
            assert isinstance(sae_cfg, SAEConfig)
            assert isinstance(train_cfg, TrainingConfig)
