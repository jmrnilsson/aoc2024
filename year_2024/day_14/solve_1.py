import math
import math
import operator
import re
import sys
from collections import Counter
from copy import deepcopy
from functools import reduce
from typing import List, Tuple, Generator

import numpy as np

from aoc.helpers import locate, build_location, read_lines
from aoc.printer import get_meta_from_fn, ANSIColors, print2
from aoc.tools import pretty


sys.setrecursionlimit(30_000)

_default_puzzle_input = "year_2024/day_01/puzzle.txt"
_default_test_input = "year_2024/day_01/test.txt"

puzzle_input = build_location(__file__, "puzzle.txt")
test_input = build_location(__file__, "test.txt")
test_input_2 = build_location(__file__, "test_2.txt")

def move(shape: Tuple[int, int], robots: List, turns: int, print: bool = False):
    k = turns
    moved_robots = []
    for n, robot in enumerate(robots):
        p0, p1, v0, v1 = robot
        # p0, p1, k, v0, v1, height, width = symbols("p0 p1 k v0 v1 height width")

        y = (p1 + k * v1) % shape[0]
        x = (p0 + k * v0) % shape[1]
        moved_robots.append((n, y, x))

    return moved_robots

def each_tick_print(shape, robots: list, ticks = 5):
    grid_ = np.zeros(shape, dtype=np.int32)

    for n, robot in enumerate(robots):
        p0, p1, v0, v1 = robot
        grid_[p1, p0] = grid_[p1, p0] + 1

    print("Initial state:\n" + pretty(grid_) + "\n")

    for tick in range(1, ticks + 1):
        moved_robots_1 = move(shape, robots, tick)
        plural = "s" if tick > 1 else ""

        grid_1 = np.zeros(shape, dtype=np.int32)

        for n, robot in enumerate(moved_robots_1):
            n, y, x = robot
            grid_1[y, x] = grid_1[y, x] + 1

        print(f"After {tick} second{plural}:\n" + pretty(grid_1) + "\n")

def solve_(__input=None):
    """
    :challenge: 12
    :expect: 223020000
    """
    lines = []
    with open(locate(__input), "r") as fp:
        for line in read_lines(fp):
            lines.append(line)

    robots = []
    for line in lines:
        robot = list(map(int, re.findall(r"-?\d+", line)))
        robots.append(robot)

    shape = (7, 11) if "test" in __input else (103, 101)
    troubleshoot = True if "test_2" in __input else False

    each_tick_print(shape, robots)

    moved_robots = move(shape, robots, 100, troubleshoot)

    _grid = np.zeros(shape, dtype=np.int32)

    totals = Counter()
    for name, y, x in moved_robots:
        if y - shape[0] // 2 == 0:
            continue

        if x - shape[1] // 2 == 0:
            continue

        _grid[y, x] = 1 + _grid[y, x]

        qy = y // ((shape[0] // 2) + 1)
        qx = x // ((shape[1] // 2) + 1)
        totals.update({(qy, qx): 1})

    print(pretty(_grid))

    return reduce(operator.mul, totals.values())


if __name__ == "__main__":
    challenge = get_meta_from_fn(solve_, "challenge")
    expect = get_meta_from_fn(solve_, "expect")
    print2(solve_, test_input, challenge)
    print2(solve_, test_input_2, challenge)
    print2(solve_, puzzle_input, expect, ANSIColors.OK_GREEN)
