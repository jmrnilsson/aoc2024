import itertools
import operator
import re
import statistics
import sys
from collections import Counter, OrderedDict, defaultdict
from copy import copy, deepcopy
from dataclasses import dataclass
from enum import Enum
from functools import reduce
from itertools import combinations
from typing import Dict, List, Callable, Tuple, Literal, Set, Generator, Any

import more_itertools
# import numba
import numpy as np
from defaultlist import defaultlist
from more_itertools import windowed, chunked
from more_itertools.recipes import sliding_window
from numpy._core._multiarray_umath import StringDType
from numpy.random import permutation

from aoc.helpers import locate, build_location, read_lines
from aoc.poll_printer import PollPrinter
from aoc.printer import get_meta_from_fn, print_, ANSIColors, print2
from aoc.tests.test_fixtures import get_challenges_from_meta
from aoc.tools import transpose
from year_2021.day_05 import direction

# from numba import cuda, np as npc, vectorize

# requires https://developer.nvidia.com/cuda-downloads CUDA toolkit. There are some sketchy versions in pip, but it's
# almost impossible to find the right versions.
# - pip install nvidia-pyindex
# - pip install nvidia-cuda-runtime-cuXX
# python -m numba -s
# print(numba.__version__)
# print(cuda.gpus)
# print(cuda.detect())

# if cuda.is_available():
#     print("CUDA is available!")
#     print("Device:", cuda.get_current_device().name)
# else:
#     print("CUDA is not available.")

sys.setrecursionlimit(30_000)

_default_puzzle_input = "year_2024/day_01/puzzle.txt"
_default_test_input = "year_2024/day_01/test.txt"

puzzle_input = build_location(__file__, "puzzle.txt")
test_input = build_location(__file__, "test.txt")
test_input_2 = build_location(__file__, "test_2.txt")
test_input_3 = build_location(__file__, "test_3.txt")
test_input_4 = build_location(__file__, "test_4.txt")
test_input_5 = build_location(__file__, "test_5.txt")


class FloodFillAutomaton:
    visited: Set[Tuple[int, int]]
    counter: Counter

    def __init__(self, grid):
        self.visited = set()
        self.grid = grid
        self.new_grid = grid = np.full(grid.shape, dtype=StringDType, fill_value=".")
        self.counter = Counter()

    def flood(self, pos: Tuple[int, int]):
        cell = self.grid[pos]
        n = self.counter.get(self.grid[pos], 0)
        name = f"{cell}{n}"
        queue = {pos}
        added = False
        while queue:
            current = queue.pop()

            if current in self.visited:
                continue

            value = None
            neighbors = [(1, 0), (-1, 0), (0, 1), (0, -1)]
            for dy, dx in neighbors:
                y = current[0] - dy
                x = current[1] - dx
                if (y, x) not in self.visited:
                    if self.grid[(y, x)] == ".":
                        continue

                    if self.grid[(y, x)] == self.grid[pos]:
                        queue.add((y, x))

            n = self.counter.get(self.grid[current], 0)
            s = f"{self.grid[current]}{n}"
            added = True
            self.new_grid[current] = value or s
            self.visited.add(current)

        if added:
            self.counter.update({cell: 1})

    def get_grid(self):
        return self.new_grid

def solve(__input=None):
    """
    :challenge: 140
    :expect: 1467094
    """
    _lines = []
    with open(locate(__input), "r") as fp:
        for __line in read_lines(fp):
            _lines.append(list(__line))

    ylen = len(_lines) + 2
    xlen = len(_lines[0]) + 2
    shape = ylen, xlen
    _grid = np.full(shape, fill_value='.', dtype=str)

    # print(grid)

    for y in range(len(_lines)):
        for x in range(len(_lines[0])):
            _grid[y + 1, x + 1] = _lines[y][x]

    fill = FloodFillAutomaton(_grid)
    positions = ((int(y), int(x)) for y, x in np.argwhere(_grid != "."))
    for pos in positions:
        fill.flood(pos)

    grid = fill.get_grid()

    faces = defaultdict(set)
    for y1, y2 in sliding_window(range(shape[0]), 2):
        for x in range(shape[1]):
            if len((faces_ := {grid[y1, x], grid[y2, x]})) > 1:
                for face in filter(lambda f: f != ".", faces_):
                    faces[face].add((y1, y2, x, 'y'))


    for x1, x2 in sliding_window(range(shape[1]), 2):
        for y in range(shape[0]):
            if len((faces_ := {grid[y, x1], grid[y, x2]})) > 1:
                for face in filter(lambda f: f != ".", faces_):
                    faces[face].add((x1, x2, y, 'x'))

    unique, counts = np.unique(grid, return_counts=True)
    areas = dict(zip(unique, counts))

    total = 0
    for k in sorted(unique):
        if k == '.':
            continue
        area = areas[k]
        perimeter = len(faces[k])
        k_local = perimeter * area
        # print(f"{k} = {k_local}")
        total += k_local

    return total

if __name__ == "__main__":
    challenge = get_meta_from_fn(solve, "challenge")
    challenge_2 = 772
    challenge_3 = 1930
    challenge_4 = (
        2 * 6 +  # C
        1 * 4 +  # D
        1 * 4 +  # B
        1 * 4 +  # Y
        7 * 15   # A
    )
    challenge_5 = (
        2 * 6 +  # C
        1 * 4 +  # D
        2 * 6 +  # B
        1 * 4 +  # Y
        1 * 4 +  # A1
        4 * 12   # A2
    )

    expect = get_meta_from_fn(solve, "expect")
    print2(solve, test_input, challenge)
    print2(solve, test_input_2, challenge_2, ANSIColors.OK_BLUE)
    print2(solve, test_input_3, challenge_3, ANSIColors.OK_GREEN)
    print2(solve, test_input_4, challenge_4, ANSIColors.OK_CYAN)
    print2(solve, test_input_5, challenge_5, ANSIColors.WARNING)
    print2(solve, puzzle_input, expect, ANSIColors.OK_GREEN)
