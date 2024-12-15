import itertools
import operator
import re
import statistics
import sys
from collections import Counter, OrderedDict, defaultdict
from copy import copy, deepcopy
from dataclasses import dataclass
from functools import reduce
from itertools import accumulate, cycle
from time import sleep
from turtledemo.sorting_animate import instructions1
from typing import Dict, List, Callable, Tuple, Literal, Set, Generator, Any, Iterable
import os
import more_itertools
# import numba
import numpy as np
import six
from defaultlist import defaultlist
from more_itertools import windowed, chunked
from more_itertools.recipes import sliding_window

from aoc.helpers import locate, build_location, read_lines
from aoc.poll_printer import PollPrinter
from aoc.printer import get_meta_from_fn, print_, ANSIColors, print2
from aoc.tests.test_fixtures import get_challenges_from_meta
from aoc.tools import transpose, pretty
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


class StorageRobotAutomaton:
    lookup: Dict[str, Tuple[int, int]]

    def __init__(self, grid):
        self.grid = grid
        self.lookup = { "^": (-1, 0), "v": (1, 0), ">": (0, 1), "<": (0, -1) }
        self.sideways_peek_lookup = { "[": (0, 1), "]": (0, -1) }

    def _peek(self, direction_: str, y0: int, x0: int) -> List[Tuple[int, int]] | str:
        # Peek(y0, x0, self.grid[y0, x0])
        dy0, dx0 = self.lookup[direction_]
        y1, x1 = dy0 + y0, dx0 + x0
        cell_value = self.grid[y1, x1]
        if more := self.sideways_peek_lookup.get(cell_value):
            dy1, dx1 = more
            return [(y1, x1), (y1 + dy1, x1 + dx1)]

        return cell_value

    def peek(self, direction_: str, y0: int, x0: int) -> List[Tuple[int, int]]:
        heap, ok, added = [(y0, x0)], [], set()

        while heap:
            y, x = heap.pop()
            if (peeked := self._peek(direction_, y, x)) == ".":
                ok.append((y, x))
                continue
            elif peeked == "#":
                return []

            if isinstance(peeked, List):
                ok.append((y, x))
                for p in (p for p in peeked if p not in added):
                    heap.append(p)
                    added.add(p)
        return ok

    def shove(self, direction_: str, n: int) -> None:
        # before_grid = deepcopy(self.grid)
        robot, = np.argwhere(self.grid == "@").tolist()
        ok = self.peek(direction_, *robot)
        seen = set()
        assigned = set()
        if len(ok) > 0:
            values = [(*p, self.grid[*p]) for p in ok]
            for y0, x0, value in values:
                seen.add((y0, x0))
                dy0, dx0 = self.lookup[direction_]
                yx0 = y0 + dy0, x0 + dx0
                self.grid[yx0] = value
                assigned.add(yx0)

            unassigned = [s for s in seen if s not in assigned]
            for u in unassigned:
                self.grid[u] = "."

        # print(f"Move {direction_} ({n}):\r\nbefore:\r\r{pretty(before_grid)}\r\n after:\r\n{pretty(self.grid)}\r\n")

    def sum_goods_positioning_system(self):
        blocks = np.argwhere(self.grid == "[")
        return sum(100 * y + x for y, x in blocks)


def solve_(__input=None):
    """
    :challenge: 9021
    :expect: 1550677
    """
    _instructions = ""
    _grid = []
    with open(locate(__input), "r") as fp:
        for line in read_lines(fp):
            if re.search(r'[<v^>]', line):
                _instructions += line
            else:
                _grid.append(list(line))

    programming = list(_instructions)

    shape = len(_grid), len(_grid[0])
    new_shape = shape[0], shape[1] * 2
    grid = np.full(new_shape, dtype=str, fill_value=".")
    for y, x in itertools.product(range(shape[0]), range(shape[1])):
        match _grid[y][x]:
            case ".":
                pass
            case "@":
                grid[y, x * 2] = "@"
            case "O":
                grid[y, x * 2] = "["
                grid[y, x * 2 + 1] = "]"
            case "#":
                grid[y, x * 2] = "#"
                grid[y, x * 2 + 1] = "#"

    storage_robot = StorageRobotAutomaton(grid)
    for n, move_direction in enumerate(programming):
        storage_robot.shove(move_direction, n)

    return storage_robot.sum_goods_positioning_system()


if __name__ == "__main__":
    challenge = get_meta_from_fn(solve_, "challenge")
    challenge_2 = 9021
    # challenge_3 = 105
    expect = get_meta_from_fn(solve_, "expect")
    # print2(solve_, test_input, challenge)
    print2(solve_, test_input_2, challenge_2, ANSIColors.OK_BLUE)
    # print2(solve_, test_input_3, challenge_3, ANSIColors.WARNING)
    print2(solve_, puzzle_input, expect, ANSIColors.OK_GREEN)
