"""
Grid Encoders for ARC Tasks

Provides various text representations for ARC grids:
- MinimalGridEncoder: Compact digit representation
- GridWithSeparationEncoder: Digits with separator characters
- GridCodeBlockEncoder: Wrapped in code blocks
- GridShapeEncoder: Includes grid dimensions
- RowNumberEncoder: Numbered rows
- RepeatNumberEncoder: Repeated digits
- ReplaceNumberEncoder: Unicode symbol substitution
"""

from abc import ABC, abstractmethod
import numpy as np
import logging

logger = logging.getLogger(__name__)


def create_grid_encoder(encoder_name: str):
    """
    Create a grid encoder from a string specification.
    
    WARNING: This uses eval() which is a security risk. Only use with trusted input.
    
    Examples:
        create_grid_encoder("MinimalGridEncoder()")
        create_grid_encoder("GridCodeBlockEncoder(MinimalGridEncoder())")
        create_grid_encoder("GridWithSeparationEncoder('|')")
    
    Args:
        encoder_name: String specification of encoder
        
    Returns:
        GridEncoder instance
        
    Raises:
        ValueError: If encoder_name doesn't create a valid GridEncoder
    """
    grid_encoder = eval(encoder_name)
    if isinstance(grid_encoder, GridEncoder):
        logger.info(f'Created `{encoder_name}` as grid encoder')
        return grid_encoder
    else:
        raise ValueError(f'{encoder_name} is not a GridEncoder subclass')


class GridEncoder(ABC):
    """Abstract base class for grid encoders."""
    
    @abstractmethod
    def to_text(self, grid) -> str:
        """Convert grid to text representation."""
        pass

    @abstractmethod
    def to_grid(self, text: str) -> list:
        """Convert text representation back to grid."""
        pass


class MinimalGridEncoder(GridEncoder):
    """
    Minimal grid encoding - just digits with newlines.
    
    Example:
        012
        345
        678
    """
    
    @staticmethod
    def to_text(grid) -> str:
        text = '\n'.join([''.join([str(x) for x in line]) for line in grid])
        return text

    @staticmethod
    def to_grid(text: str) -> list:
        lines = text.strip().splitlines()
        grid = [[int(x) for x in line] for line in lines]
        return grid


class GridWithSeparationEncoder(GridEncoder):
    """
    Grid encoding with separator between cells.
    
    Example (with '|' separator):
        0|1|2
        3|4|5
        6|7|8
    """
    
    def __init__(self, split_symbol: str):
        self.split_symbol = split_symbol

    def to_text(self, grid) -> str:
        text = '\n'.join([self.split_symbol.join([str(x) for x in line]) for line in grid])
        return text

    def to_grid(self, text: str) -> list:
        lines = text.strip().splitlines()
        grid = [[int(x) for x in line.split(self.split_symbol)] for line in lines]
        return grid


class GridCodeBlockEncoder(GridEncoder):
    """
    Wraps grid in markdown-style code block.
    
    Example:
        ```grid
        012
        345
        678
        ```
    """
    
    def __init__(self, base_encoder: GridEncoder):
        self.encoder = base_encoder

    def to_text(self, grid) -> str:
        text = f'```grid\n{self.encoder.to_text(grid)}\n```'
        return text

    def to_grid(self, text: str) -> list:
        grid_text = text.split('```grid\n')[1].split('\n```')[0]
        grid = self.encoder.to_grid(grid_text)
        return grid


class GridShapeEncoder(GridEncoder):
    """
    Includes grid shape in the encoding.
    
    Example:
        ```grid shape: 3x3
        012
        345
        678
        ```
    """
    
    def __init__(self, base_encoder: GridEncoder):
        self.encoder = base_encoder

    def to_text(self, grid) -> str:
        text = f'```grid shape: {len(grid)}x{len(grid[0])}\n{self.encoder.to_text(grid)}\n```'
        return text

    def to_grid(self, text: str) -> list:
        grid_lines = []
        is_grid_line = False
        for line in text.splitlines():
            if line.startswith('```grid shape:'):
                is_grid_line = True
            elif is_grid_line:
                if line.startswith('```'):
                    break
                grid_lines.append(line)
        grid_text = '\n'.join(grid_lines)
        grid = self.encoder.to_grid(grid_text)
        return grid


class RowNumberEncoder(GridEncoder):
    """
    Adds row numbers to each line.
    
    Example:
        1 012
        2 345
        3 678
    """
    
    def __init__(self, base_encoder: GridEncoder):
        self.encoder = base_encoder

    def to_text(self, grid) -> str:
        text = self.encoder.to_text(grid)
        text_with_row_numbers = ''
        for idx, line in enumerate(text.splitlines()):
            text_with_row_numbers += f'{idx+1} {line}\n'
        return text_with_row_numbers.strip()

    def to_grid(self, text: str) -> list:
        text_without_row_numbers = ''
        for line in text.splitlines():
            text_without_row_numbers += line.split(' ', 1)[1] + '\n'
        grid = self.encoder.to_grid(text_without_row_numbers)
        return grid


class RepeatNumberEncoder(GridEncoder):
    """
    Repeats each digit n times for better tokenization.
    
    Example (n=3):
        000111222
        333444555
        666777888
    """
    
    def __init__(self, n: int = 3):
        self.n = n

    def to_text(self, grid) -> str:
        text = '\n'.join([''.join([str(x)*self.n for x in line]) for line in grid])
        return text

    def to_grid(self, text: str) -> list:
        lines = text.strip().splitlines()
        grid = [[int(x) for x in line[::self.n]] for line in lines]
        return grid


class ReplaceNumberEncoder(GridEncoder):
    """
    Replaces digits with Unicode symbols for better tokenization.
    
    Uses unique single-token characters for each digit 0-9.
    """
    
    symbols = ['ñ', 'ò', '÷', 'û', 'ą', 'ć', 'ď', 'ę', 'Ě', 'Ğ']

    def __init__(self, base_encoder: GridEncoder):
        self.encoder = base_encoder

    def to_text(self, grid) -> str:
        text = self.encoder.to_text(grid)
        for idx, symbol in enumerate(self.symbols):
            text = text.replace(str(idx), symbol)
        return text

    def to_grid(self, text: str) -> list:
        for idx, symbol in enumerate(self.symbols):
            text = text.replace(symbol, str(idx))
        grid = self.encoder.to_grid(text)
        return grid


def test_grid_encoder_is_reversible(encoder_name: str) -> bool:
    """
    Test that an encoder can round-trip a grid.
    
    Args:
        encoder_name: Encoder specification string
        
    Returns:
        True if reversible
    """
    grid_encoder = create_grid_encoder(encoder_name)
    sample_grid = np.reshape(np.arange(9), (3, 3)).tolist()
    result = grid_encoder.to_grid(grid_encoder.to_text(sample_grid)) == sample_grid
    if result:
        print(f"✓ {encoder_name} is reversible")
    else:
        print(f"✗ {encoder_name} is NOT reversible")
    return result
