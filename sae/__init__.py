"""Generic Sparse Autoencoder (SAE) training library for JAX/TPU."""

__version__ = "0.1.0"

from .configs.base import SAEConfig
from .configs.training import TrainingConfig
from .models.base import BaseSAE
from .models.vanilla import VanillaSAE
from .models.topk import TopKSAE
from .models.gated import GatedSAE
from .models.jumprelu import JumpReLUSAE
from .models.registry import create_sae, register_sae
from .data.base import ActivationSource
from .data.registry import create_source, register_source
from .training.trainer import SAETrainer
