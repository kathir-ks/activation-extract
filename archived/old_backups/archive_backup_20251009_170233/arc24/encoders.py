from abc import ABC, abstractmethod
import numpy as np
import logging

logger = logging.getLogger(__name__)


def create_grid_encoder(encoder_name):
    """
    This is a security risk, but I'm the only user of the library
    Otherwise I will need to write and maintain dictionary with the encoder names
    It allows me to use plain text in the configuration to specify the encoder

    Examples of encoders:

    GridCodeBlockEncoder(MinimalGridEncoder())
    GridCodeBlockEncoder(GridWithSeparationEncoder('|'))
    """
    grid_encoder = eval(encoder_name)
    if isinstance(grid_encoder, GridEncoder):
        logger.info(f'Created `{encoder_name}` as grid encoder')
        return grid_encoder
    else:
        raise ValueError(f'{encoder_name} is not a GridEncoder subclass')


class GridEncoder(ABC):
    @abstractmethod
    def to_text(self, grid):
        pass

    @abstractmethod
    def to_grid(self, text):
        pass


class MinimalGridEncoder(GridEncoder):
    @staticmethod
    def to_text(grid):
        text = '\n'.join([''.join([str(x) for x in line]) for line in grid])
        return text

    @staticmethod
    def to_grid(text):
        lines = text.strip().splitlines()
        grid = [[int(x) for x in line] for line in lines]
        return grid


class GridWithSeparationEncoder(GridEncoder):
    def __init__(self, split_symbol):
        self.split_symbol = split_symbol

    def to_text(self, grid):
        text = '\n'.join([self.split_symbol.join([str(x) for x in line]) for line in grid])
        return text

    def to_grid(self, text):
        lines = text.strip().splitlines()
        grid = [[int(x) for x in line.split(self.split_symbol)] for line in lines]
        return grid


class GridCodeBlockEncoder(GridEncoder):
    def __init__(self, base_encoder):
        self.encoder = base_encoder

    def to_text(self, grid):
        text = f'```grid\n{self.encoder.to_text(grid)}\n```'
        return text

    def to_grid(self, text):
        grid_text = text.split('```grid\n')[1].split('\n```')[0]
        grid = self.encoder.to_grid(grid_text)
        return grid


class GridShapeEncoder(GridEncoder):
    def __init__(self, base_encoder):
        self.encoder = base_encoder

    def to_text(self, grid):
        text = f'```grid shape: {len(grid)}x{len(grid[0])}\n{self.encoder.to_text(grid)}\n```'
        return text

    def to_grid(self, text):
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
    def __init__(self, base_encoder):
        self.encoder = base_encoder

    def to_text(self, grid):
        text = self.encoder.to_text(grid)
        text_with_row_numbers = ''
        for idx, line in enumerate(text.splitlines()):
            text_with_row_numbers += f'{idx+1} {line}\n'
        return text_with_row_numbers.strip()

    def to_grid(self, text):
        text_without_row_numbers = ''
        for line in text.splitlines():
            text_without_row_numbers += line.split(' ', 1)[1] + '\n'
        grid = self.encoder.to_grid(text_without_row_numbers)
        return grid


class RepeatNumberEncoder(GridEncoder):
    def __init__(self, n=3):
        self.n = n

    def to_text(self, grid):
        text = '\n'.join([''.join([str(x)*self.n for x in line]) for line in grid])
        return text

    def to_grid(self, text):
        lines = text.strip().splitlines()
        #TODO: make something more robust
        grid = [[int(x) for x in line[::self.n]] for line in lines]
        return grid


class ReplaceNumberEncoder(GridEncoder):
    symbols = ['ñ', 'ò', '÷', 'û', 'ą', 'ć', 'ď', 'ę', 'Ě', 'Ğ']

    def __init__(self, base_encoder):
        self.encoder = base_encoder

    def to_text(self, grid):
        text = self.encoder.to_text(grid)
        for idx, symbol in enumerate(self.symbols):
            text = text.replace(str(idx), symbol)
        return text

    def to_grid(self, text):
        for idx, symbol in enumerate(self.symbols):
            text = text.replace(symbol, str(idx))
        grid = self.encoder.to_grid(text)
        return grid


def test_grid_encoder_is_reversible(encoder_name):
    grid_encoder = create_grid_encoder(encoder_name)
    sample_grid = np.eye(3, dtype=int).tolist()
    sample_grid = np.reshape(np.arange(9), (3, 3)).tolist()
    assert sample_grid == grid_encoder.to_grid(grid_encoder.to_text(sample_grid))
    print(grid_encoder.to_text(sample_grid) + '\n')


if __name__ == '__main__':
    test_grid_encoder_is_reversible('MinimalGridEncoder()')
    test_grid_encoder_is_reversible("GridWithSeparationEncoder('|')")
    test_grid_encoder_is_reversible('GridCodeBlockEncoder(MinimalGridEncoder())')
    test_grid_encoder_is_reversible("GridCodeBlockEncoder(GridWithSeparationEncoder('|'))")
    test_grid_encoder_is_reversible('GridCodeBlockEncoder(RepeatNumberEncoder(3))')
    test_grid_encoder_is_reversible('GridCodeBlockEncoder(RepeatNumberEncoder(2))')
    test_grid_encoder_is_reversible('GridShapeEncoder(MinimalGridEncoder())')
    test_grid_encoder_is_reversible('GridShapeEncoder(RepeatNumberEncoder(3))')
    test_grid_encoder_is_reversible('GridCodeBlockEncoder(RowNumberEncoder(MinimalGridEncoder()))')
    test_grid_encoder_is_reversible('GridShapeEncoder(RowNumberEncoder(MinimalGridEncoder()))')
    test_grid_encoder_is_reversible('GridShapeEncoder(ReplaceNumberEncoder(MinimalGridEncoder()))')
    test_grid_encoder_is_reversible('GridShapeEncoder(RowNumberEncoder(ReplaceNumberEncoder(MinimalGridEncoder())))')
