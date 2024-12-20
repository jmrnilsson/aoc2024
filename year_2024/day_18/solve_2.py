import heapq
import itertools
import operator
import re
import statistics
import sys
import traceback
from bisect import bisect
from collections import Counter, OrderedDict, defaultdict
from copy import copy, deepcopy
from dataclasses import dataclass
from functools import reduce
from typing import Dict, List, Callable, Tuple, Literal, Set, Generator, Any

import more_itertools
# import numba
import numpy as np
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
test_input_4 = build_location(__file__, "test_4.txt")
test_input_5 = build_location(__file__, "test_5.txt")


def get_neighbors(node, grid):
    von_neumann = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    neighbors = []

    for dy, dx in von_neumann:
        neighbor = node[0] + dy, node[1] + dx

        if -1 < neighbor[0] < grid.shape[0] and -1 < neighbor[1] < grid.shape[1]:
            if grid[neighbor] != "#":
                neighbors.append(neighbor)

    return neighbors

def dijkstra(grid, start, end, costs):
    seen = set()
    q = [start]
    costs[start] = 0
    # heapq.heapify(pq)
    while q:
        current = q.pop(0)

        if current == end:
            break

        if current in seen:
            continue

        seen.add(current)

        for n in get_neighbors(current, grid):
            neighbor_cost = costs[n]
            current_cost = costs[current]
            if neighbor_cost > current_cost + 1:
                costs[n] = current_cost + 1
                if n in seen:
                    seen.remove(n)

            q.append(n)

def run_dijkstra(coords, bytes_: int, shape, costs):
    grid = np.full(shape, dtype=str, fill_value=".")
    for y, x in coords[:bytes_]:
        grid[y, x] = "#"

    start = (0, 0)
    end = (shape[0] - 1, shape[1] - 1)
    dijkstra(grid, start, end, costs)

def solve_(__input=None):
    """
    :challenge: 6,1
    :expect: 26,50
    """
    coords = []
    with open(locate(__input), "r") as fp:
        for line in read_lines(fp):
            x, y = line.split(",")
            coords.append((int(y), int(x)))


    if "test" in  __input:
        shape = (7, 7)
        bytes_ = 12
    else:
        shape = (71, 71)
        bytes_ = 1024

    max_bytes = len(coords)

    start = (0, 0)
    end = (shape[0] - 1, shape[1] - 1)

    # Classic bisect w/o built-in Python variant. Learn this through.
    min_ = bytes_
    max_ = max_bytes
    while 1:
        costs = np.full(shape, dtype=int, fill_value=sys.maxsize)
        delta = max_ - min_

        if delta < 2:
            y, x = coords[min_]
            return f"{x},{y}"

        bisect_at = min_ + delta // 2
        run_dijkstra(coords, bisect_at, shape, costs)

        if costs[end] == sys.maxsize:
            max_ = bisect_at
        else:
            min_ = bisect_at

if __name__ == "__main__":
    challenge = get_meta_from_fn(solve_, "challenge")
    expect = get_meta_from_fn(solve_, "expect")
    print2(solve_, test_input, challenge)
    print2(solve_, puzzle_input, expect, ANSIColors.OK_GREEN)
