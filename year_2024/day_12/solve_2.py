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
from aoc.tools import transpose, group_by
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
test_input_6 = build_location(__file__, "test_6.txt")
test_input_7 = build_location(__file__, "test_7.txt")


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

def _solve(__input=None):
    """
    :challenge: 80
    :expect: 881182
    """
    _lines = []
    with open(locate(__input), "r") as fp:
        for __line in read_lines(fp):
            _lines.append(list(__line))

    height = len(_lines) + 2
    width = len(_lines[0]) + 2
    shape = height, width
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
                    association = 'up' if grid[y1, x] == face else 'down'
                    faces[face].add((y1, y2, x, 'y', association))


    for x1, x2 in sliding_window(range(shape[0]), 2):
        for y in range(shape[0]):
            if len((faces_ := {grid[y, x1], grid[y, x2]})) > 1:
                for face in filter(lambda f: f != ".", faces_):
                    association = 'left' if grid[y, x1] == face else 'right'
                    faces[face].add((x1, x2, y, 'x', association))

    for name, faces_ in faces.items():
        mat = list(filter(lambda r: r[3] == "y", faces_))
        iterable = sorted(mat, key=lambda t: (t[0], t[1], t[2]))
        sw = list(more_itertools.sliding_window(iterable, 2))
        for a, b in sw:
            # if (a[0], a[1]) == (b[0], b[1]) and abs(a[2] - b[2]) < 2 and (grid[a[0], a[2]], grid[a[1], a[2]]) == (grid[b[0], b[2]], grid[b[1], b[2]]):
            one = a[-1]
            two = b[-1]
            if (a[0], a[1]) == (b[0], b[1]) and abs(a[2] - b[2]) < 2 and one == two:
                faces[name].remove(a)

    for name, faces_ in faces.items():
        mat = list(filter(lambda r: r[3] == "x", faces_))
        iterable = sorted(mat, key=lambda t: (t[0], t[1], t[2]))
        sw = list(more_itertools.sliding_window(iterable, 2))
        for a, b in sw:
            # if (a[0], a[1]) == (b[0], b[1]) and abs(a[2] - b[2]) < 2 and (grid[a[0], a[2]], grid[a[1], a[2]]) == (grid[b[0], b[2]], grid[b[1], b[2]]):
            if (a[0], a[1]) == (b[0], b[1]) and abs(a[2] - b[2]) < 2 and a[-1] == b[-1]:
                faces[name].remove(a)

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
    challenge = get_meta_from_fn(_solve, "challenge")
    challenge_2 = get_meta_from_fn(_solve, "challenge_2")
    challenge_3 = get_meta_from_fn(_solve, "challenge_3")
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

    challenge = get_meta_from_fn(_solve, "challenge")
    challenge_2 = get_meta_from_fn(_solve, "challenge_2")
    expect = get_meta_from_fn(_solve, "expect")

    print2(_solve, test_input, challenge)
    print2(_solve, test_input_2, 436, ANSIColors.OK_BLUE)
    print2(_solve, test_input_6, 236, ANSIColors.OK_GREEN)
    print2(_solve, test_input_7, 368, ANSIColors.WARNING)
    print2(_solve, test_input, 1930, ANSIColors.WARNING)
    print2(_solve, puzzle_input, expect, ANSIColors.OK_GREEN)
