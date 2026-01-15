"""
Extraction Module

Provides utilities for activation extraction pipelines:
- Activation storage with automatic sharding
- Single-host extraction
- Multi-host TPU extraction
"""

from .storage import (
    ActivationStorage,
    load_activation_shard,
)

from .extractor import (
    ExtractionConfig,
    run_extraction,
    run_extraction_single_host,
    run_extraction_multi_host,
)


__all__ = [
    # Storage
    'ActivationStorage',
    'load_activation_shard',
    
    # Extractor
    'ExtractionConfig',
    'run_extraction',
    'run_extraction_single_host',
    'run_extraction_multi_host',
]
