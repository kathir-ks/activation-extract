"""
Data Augmentation for ARC Tasks

Provides various augmentation strategies:
- Geometric: rotation, horizontal flip
- Color: random color permutation
- Padding: add border padding
- Upscale: repeat pixels
- Mirror: duplicate along axis

These augmentations help increase training data variety while
preserving the underlying task logic.
"""

import numpy as np
import random
from functools import partial
from typing import Dict, List, Tuple, Optional

MAX_GRID_SIZE = 30


class GridTooBigToAugmentError(Exception):
    """Raised when a grid is too large for the requested augmentation."""
    pass


def set_random_seed(random_seed: int) -> None:
    """Set random seed for reproducibility."""
    random.seed(random_seed)
    np.random.seed(random_seed)


# =============================================================================
# Core Augmentation Functions
# =============================================================================

def apply_data_augmentation(
    task: Dict, 
    hflip: bool, 
    n_rot90: int, 
    color_map: Optional[Dict[int, int]] = None
) -> Dict:
    """
    Apply geometric augmentation and optional color swapping to a task.
    
    Args:
        task: Task dict with 'train' and 'test' keys
        hflip: Whether to apply horizontal flip
        n_rot90: Number of 90-degree rotations (0-3)
        color_map: Optional color remapping dict {old: new}
        
    Returns:
        Augmented task dict
    """
    augmented_task = _apply_augmentation_to_task(
        task, 
        partial(geometric_augmentation, hflip=hflip, n_rot90=n_rot90)
    )
    if color_map is not None:
        augmented_task = swap_task_colors(augmented_task, color_map)
    augmented_task = permute_train_samples(augmented_task)
    return augmented_task


def revert_data_augmentation(
    grid: List[List[int]], 
    hflip: bool, 
    n_rot90: int, 
    color_map: Optional[Dict[int, int]] = None
) -> List[List[int]]:
    """
    Revert augmentation on a grid (for predictions).
    
    Args:
        grid: Augmented grid
        hflip: Whether hflip was applied
        n_rot90: Number of 90-degree rotations that were applied
        color_map: Color map that was used
        
    Returns:
        Original grid
    """
    grid = revert_geometric_augmentation(grid, hflip, n_rot90)
    if color_map is not None:
        grid = revert_color_swap(grid, color_map)
    return grid


def random_augment_task(task: Dict, swap_train_and_test: bool = True) -> Dict:
    """
    Apply random augmentations to a task.
    
    Args:
        task: Task dict
        swap_train_and_test: Whether to potentially swap train/test samples
        
    Returns:
        Randomly augmented task
    """
    augmented_task = apply_data_augmentation(
        task, 
        color_map=get_random_color_map(), 
        **get_random_geometric_augmentation_params()
    )
    if swap_train_and_test:
        augmented_task = random_swap_train_and_test(augmented_task)
    return augmented_task


# =============================================================================
# Geometric Augmentation
# =============================================================================

def get_random_geometric_augmentation_params() -> Dict:
    """Get random geometric augmentation parameters."""
    return dict(hflip=random.choice([True, False]), n_rot90=random.choice([0, 1, 2, 3]))


def geometric_augmentation(grid: List[List[int]], hflip: bool, n_rot90: int) -> List[List[int]]:
    """
    Apply geometric transformations to a grid.
    
    Args:
        grid: Input grid
        hflip: Whether to flip horizontally
        n_rot90: Number of 90-degree counter-clockwise rotations
        
    Returns:
        Transformed grid
    """
    grid = np.array(grid)
    if hflip:
        grid = np.flip(grid, axis=1)
    grid = np.rot90(grid, k=n_rot90)
    return grid.tolist()


def revert_geometric_augmentation(grid: List[List[int]], hflip: bool, n_rot90: int) -> List[List[int]]:
    """Revert geometric transformations."""
    grid = np.array(grid)
    grid = np.rot90(grid, k=-n_rot90)
    if hflip:
        grid = np.flip(grid, axis=1)
    return grid.tolist()


# =============================================================================
# Color Augmentation
# =============================================================================

def get_random_color_map(change_background_probability: float = 0.1) -> Dict[int, int]:
    """
    Get a random color remapping.
    
    Args:
        change_background_probability: Probability of changing color 0 (background)
        
    Returns:
        Dict mapping old colors to new colors
    """
    colors = list(range(10))
    if random.random() < change_background_probability:
        new_colors = list(range(10))
        random.shuffle(new_colors)
    else:
        new_colors = list(range(1, 10))
        random.shuffle(new_colors)
        new_colors = [0] + new_colors

    color_map = {x: y for x, y in zip(colors, new_colors)}
    return color_map


def swap_task_colors(
    task: Dict, 
    color_map: Optional[Dict[int, int]] = None,
    change_background_probability: float = 0.1
) -> Dict:
    """
    Swap colors in a task according to color map.
    
    Args:
        task: Task dict
        color_map: Color remapping dict. If None, creates random one.
        change_background_probability: For random color map
        
    Returns:
        Task with swapped colors
    """
    if color_map is None:
        color_map = get_random_color_map(change_background_probability)
    vectorized_mapping = np.vectorize(color_map.get)

    new_task = dict()
    for key in task.keys():
        new_task[key] = [
            {name: vectorized_mapping(grid).tolist() for name, grid in sample.items()} 
            for sample in task[key]
        ]
    return new_task


def revert_color_swap(grid: List[List[int]], color_map: Dict[int, int]) -> List[List[int]]:
    """Revert color swap using inverse mapping."""
    reverse_color_map = {v: k for k, v in color_map.items()}
    vectorized_mapping = np.vectorize(reverse_color_map.get)
    return vectorized_mapping(grid).tolist()


# =============================================================================
# Sample Permutation
# =============================================================================

def permute_train_samples(task: Dict) -> Dict:
    """Randomly reorder training samples."""
    train_order = np.arange(len(task['train']))
    np.random.shuffle(train_order)
    augmented_task = dict()
    augmented_task['train'] = [task['train'][idx] for idx in train_order]
    augmented_task['test'] = task['test']
    return augmented_task


def random_swap_train_and_test(task: Dict) -> Dict:
    """Randomly redistribute samples between train and test."""
    augmented_task = task.copy()
    all_samples = task['train'] + task['test']
    random.shuffle(all_samples)
    train_size = len(task['train'])
    augmented_task['train'] = all_samples[:train_size]
    augmented_task['test'] = all_samples[train_size:]
    return augmented_task


# =============================================================================
# Padding Augmentation
# =============================================================================

def add_padding(grid: List[List[int]], color: int, size: Tuple[int, int]) -> List[List[int]]:
    """
    Add padding around a grid.
    
    Args:
        grid: Input grid
        color: Padding color
        size: (top/bottom padding, left/right padding)
        
    Returns:
        Padded grid
    """
    cols = len(grid[0])
    padded_grid = [[color]*(cols + size[1]*2) for _ in range(size[0])]
    for row in grid:
        padded_grid.append([color]*size[1] + row + [color]*size[1])
    padded_grid += [[color]*(cols + size[1]*2) for _ in range(size[0])]
    return padded_grid


def get_random_padding_params(
    max_grid_shape: Tuple[int, int], 
    same_size_probability: float = 0.5,
    max_padding: int = 5,
    n_tries: int = 10
) -> Dict:
    """Get random padding parameters that won't exceed MAX_GRID_SIZE."""
    safe_max_padding = (
        min(MAX_GRID_SIZE - max_grid_shape[0], max_padding),
        min(MAX_GRID_SIZE - max_grid_shape[1], max_padding)
    )
    if random.random() < same_size_probability:
        safe_max_padding_min = min(safe_max_padding)
        if safe_max_padding_min < 1:
            raise GridTooBigToAugmentError(f"Grid is too big to pad: {max_grid_shape}")
        size = random.randint(1, safe_max_padding_min)
        size = (size, size)
    else:
        if min(safe_max_padding) < 1:
            raise GridTooBigToAugmentError(f"Grid is too big to pad: {max_grid_shape}")
        for _ in range(n_tries):
            size = (random.randint(1, safe_max_padding[0]), random.randint(1, safe_max_padding[1]))
            if size[0] != size[1]:
                break
    color = random.randint(0, 9)
    return dict(color=color, size=size)


# =============================================================================
# Upscale Augmentation
# =============================================================================

def upscale(grid: List[List[int]], scale: Tuple[int, int]) -> List[List[int]]:
    """
    Upscale grid by repeating pixels.
    
    Args:
        grid: Input grid
        scale: (row scale, column scale)
        
    Returns:
        Upscaled grid
    """
    grid = np.array(grid, dtype=int)
    for axis, s in enumerate(scale):
        grid = np.repeat(grid, s, axis=axis)
    return grid.tolist()


def get_random_upscale_params(
    max_grid_shape: Tuple[int, int],
    min_upscale: int = 2,
    max_upscale: int = 4,
    same_upscale_probability: float = 0.5,
    n_tries: int = 10
) -> Dict:
    """Get random upscale parameters that won't exceed MAX_GRID_SIZE."""
    safe_max_upscale = (
        min(MAX_GRID_SIZE // max_grid_shape[0], max_upscale),
        min(MAX_GRID_SIZE // max_grid_shape[1], max_upscale)
    )
    if random.random() < same_upscale_probability:
        safe_max = min(safe_max_upscale)
        if safe_max < 2:
            raise GridTooBigToAugmentError(f"Grid is too big to upscale: {max_grid_shape}")
        scale = random.randint(min_upscale, safe_max)
        return dict(scale=(scale, scale))
    else:
        if max(safe_max_upscale) < 2 or min(safe_max_upscale) < 1:
            raise GridTooBigToAugmentError(f"Grid is too big to upscale: {max_grid_shape}")
        min_upscale = 1
        for _ in range(n_tries):
            scale = (
                random.randint(min_upscale, safe_max_upscale[0]),
                random.randint(min_upscale, safe_max_upscale[1])
            )
            if scale[0] != scale[1]:
                break
        return dict(scale=scale)


# =============================================================================
# Mirror Augmentation
# =============================================================================

def mirror(grid: List[List[int]], axis: int, position: int) -> List[List[int]]:
    """
    Mirror grid along an axis.
    
    Args:
        grid: Input grid
        axis: 0 for vertical, 1 for horizontal
        position: 0 for prepend, 1 for append
        
    Returns:
        Mirrored grid
    """
    if axis == 0:
        if position == 0:
            return grid[::-1] + grid
        else:
            return grid + grid[::-1]
    elif axis == 1:
        new_grid = []
        for row in grid:
            if position == 0:
                new_grid.append(row[::-1] + row)
            else:
                new_grid.append(row + row[::-1])
        return new_grid


def get_random_mirror_params(max_grid_shape: Tuple[int, int]) -> Dict:
    """Get random mirror parameters that won't exceed MAX_GRID_SIZE."""
    if MAX_GRID_SIZE // max_grid_shape[0] < 2:
        if MAX_GRID_SIZE // max_grid_shape[1] < 2:
            raise GridTooBigToAugmentError(f"Grid is too big to mirror: {max_grid_shape}")
        else:
            axis = 1
    elif MAX_GRID_SIZE // max_grid_shape[1] < 2:
        axis = 0
    else:
        axis = random.randint(0, 1)
    return dict(axis=axis, position=random.randint(0, 1))


# =============================================================================
# Helper Functions
# =============================================================================

def _apply_augmentation_to_task(task: Dict, augmentation, augmentation_targets=None) -> Dict:
    """Apply augmentation function to all samples in a task."""
    augmented_task = dict()
    for partition, samples in task.items():
        augmented_task[partition] = [
            _augment_sample(sample, augmentation, augmentation_targets) 
            for sample in samples
        ]
    return augmented_task


def _augment_sample(sample: Dict, augmentation, augmentation_targets=None) -> Dict:
    """Apply augmentation to a single sample."""
    if augmentation_targets is None:
        return {name: augmentation(grid) for name, grid in sample.items()}
    else:
        if all(target not in sample for target in augmentation_targets):
            raise ValueError(f"augmentation_target {augmentation_targets} not found in sample")
        return {
            name: augmentation(grid) if name in augmentation_targets else grid 
            for name, grid in sample.items()
        }


def get_max_grid_shape(task: Dict, augmentation_targets: List[str]) -> Tuple[int, int]:
    """Get maximum grid dimensions in a task for specific targets."""
    max_shape = (0, 0)
    for _, samples in task.items():
        for sample in samples:
            for target in augmentation_targets:
                if target in sample:
                    grid = sample[target]
                    max_shape = (max(max_shape[0], len(grid)), max(max_shape[1], len(grid[0])))
    return max_shape
