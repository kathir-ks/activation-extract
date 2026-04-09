"""Model registry for SAE architectures."""

from ..configs.base import SAEConfig
from .base import BaseSAE
from .vanilla import VanillaSAE
from .topk import TopKSAE
from .gated import GatedSAE
from .jumprelu import JumpReLUSAE

SAE_REGISTRY = {
    "vanilla": VanillaSAE,
    "topk": TopKSAE,
    "gated": GatedSAE,
    "jumprelu": JumpReLUSAE,
}


def create_sae(config: SAEConfig) -> BaseSAE:
    """Create an SAE model from config."""
    if config.architecture not in SAE_REGISTRY:
        raise ValueError(
            f"Unknown architecture: {config.architecture}. "
            f"Available: {list(SAE_REGISTRY.keys())}"
        )
    cls = SAE_REGISTRY[config.architecture]
    return cls(config=config)


def register_sae(name: str, cls: type):
    """Register a custom SAE architecture."""
    SAE_REGISTRY[name] = cls
