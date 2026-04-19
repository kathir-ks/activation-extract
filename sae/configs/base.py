"""SAE model configuration."""

from dataclasses import dataclass


@dataclass
class SAEConfig:
    # Input dimension (must match source model's hidden_dim)
    hidden_dim: int = 896

    # Dictionary size (number of SAE features)
    # Common expansion factors: 8x, 16x, 32x, 64x
    dict_size: int = 896 * 16  # 14336

    # Architecture: "vanilla", "topk", "gated", "jumprelu"
    architecture: str = "vanilla"

    # -- Sparsity parameters (architecture-specific) --
    # Sparsity penalty coefficient: L1 (vanilla/gated) or L0 (jumprelu)
    l1_coeff: float = 5e-3

    # TopK: number of active features
    k: int = 64

    # TopK: auxiliary loss coefficient for dead feature gradients
    topk_aux_coeff: float = 1.0 / 32

    # JumpReLU: initial threshold and STE bandwidth
    jumprelu_init_threshold: float = 0.001
    jumprelu_bandwidth: float = 0.001

    # GatedSAE: Heaviside STE bandwidth
    gated_bandwidth: float = 0.001

    # -- Initialization --
    dec_init_norm: float = 0.1

    # -- Constraints --
    normalize_decoder: bool = True

    # -- Dtype --
    # "float32" or "bfloat16"
    dtype: str = "bfloat16"
