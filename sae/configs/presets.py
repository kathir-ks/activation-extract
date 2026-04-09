"""Pre-built configurations for common setups."""

from .base import SAEConfig
from .training import TrainingConfig


def qwen_0_5b_vanilla(layer: int = 12, expansion: int = 16) -> tuple:
    """Vanilla SAE for Qwen 2.5-0.5B (hidden_dim=896, 24 layers)."""
    sae_cfg = SAEConfig(
        hidden_dim=896,
        dict_size=896 * expansion,
        architecture="vanilla",
        l1_coeff=5e-3,
    )
    train_cfg = TrainingConfig(
        layer_index=layer,
        batch_size=4096,
        num_steps=100_000,
        learning_rate=3e-4,
    )
    return sae_cfg, train_cfg


def qwen_0_5b_topk(layer: int = 12, expansion: int = 16, k: int = 64) -> tuple:
    """TopK SAE for Qwen 2.5-0.5B."""
    sae_cfg = SAEConfig(
        hidden_dim=896,
        dict_size=896 * expansion,
        architecture="topk",
        k=k,
    )
    train_cfg = TrainingConfig(
        layer_index=layer,
        batch_size=4096,
        num_steps=100_000,
        learning_rate=3e-4,
    )
    return sae_cfg, train_cfg


PRESETS = {
    "qwen-0.5b-vanilla": qwen_0_5b_vanilla,
    "qwen-0.5b-topk": qwen_0_5b_topk,
}
