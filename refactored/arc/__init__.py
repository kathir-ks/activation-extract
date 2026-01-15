"""
ARC-AGI Pipeline Package

This module provides tools for working with ARC-AGI (Abstraction and Reasoning Corpus)
format data for activation extraction and model training.

Components:
- encoders: Grid encoding/decoding for different text representations
- prompting: Prompt template generation for different task types
- augmentation: Data augmentation for ARC tasks
"""

# Grid encoders
from .encoders import (
    GridEncoder,
    MinimalGridEncoder,
    GridWithSeparationEncoder,
    GridCodeBlockEncoder,
    GridShapeEncoder,
    RowNumberEncoder,
    RepeatNumberEncoder,
    ReplaceNumberEncoder,
    create_grid_encoder,
)

# Prompting utilities
from .prompting import (
    create_prompts_from_task,
    get_prompt_templates,
    parse_grid_from_response,
    remove_assistant_ending,
    print_smallest_prompt,
    pretty_print_prompt,
)

# Data augmentation
from .augmentation import (
    apply_data_augmentation,
    revert_data_augmentation,
    random_augment_task,
    geometric_augmentation,
    revert_geometric_augmentation,
    get_random_geometric_augmentation_params,
    get_random_color_map,
    swap_task_colors,
    revert_color_swap,
    permute_train_samples,
    random_swap_train_and_test,
    set_random_seed,
    add_padding,
    upscale,
    mirror,
    GridTooBigToAugmentError,
    MAX_GRID_SIZE,
)

__all__ = [
    # Encoders
    'GridEncoder',
    'MinimalGridEncoder',
    'GridWithSeparationEncoder',
    'GridCodeBlockEncoder',
    'GridShapeEncoder',
    'RowNumberEncoder',
    'RepeatNumberEncoder',
    'ReplaceNumberEncoder',
    'create_grid_encoder',
    
    # Prompting
    'create_prompts_from_task',
    'get_prompt_templates',
    'parse_grid_from_response',
    'remove_assistant_ending',
    'print_smallest_prompt',
    'pretty_print_prompt',
    
    # Augmentation
    'apply_data_augmentation',
    'revert_data_augmentation',
    'random_augment_task',
    'geometric_augmentation',
    'revert_geometric_augmentation',
    'get_random_geometric_augmentation_params',
    'get_random_color_map',
    'swap_task_colors',
    'revert_color_swap',
    'permute_train_samples',
    'random_swap_train_and_test',
    'set_random_seed',
    'add_padding',
    'upscale',
    'mirror',
    'GridTooBigToAugmentError',
    'MAX_GRID_SIZE',
]
