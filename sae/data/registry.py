"""Data source registry."""

from typing import Dict, Type

from .base import ActivationSource
from .numpy_source import NumpySource
from .pickle_source import PickleShardSource
from .safetensors_source import SafetensorsSource
from .hf_dataset_source import HFDatasetSource

SOURCE_REGISTRY: Dict[str, Type[ActivationSource]] = {
    "numpy": NumpySource,
    "pickle": PickleShardSource,
    "safetensors": SafetensorsSource,
    "hf_dataset": HFDatasetSource,
}


def create_source(source_type: str, **kwargs) -> ActivationSource:
    """Create an activation source by name.

    Args:
        source_type: Registered source name.
        **kwargs: Passed to the source constructor.
    """
    if source_type not in SOURCE_REGISTRY:
        raise ValueError(
            f"Unknown source: {source_type}. Available: {list(SOURCE_REGISTRY.keys())}"
        )
    return SOURCE_REGISTRY[source_type](**kwargs)


def register_source(name: str, cls: Type[ActivationSource]):
    """Register a custom activation source."""
    SOURCE_REGISTRY[name] = cls
