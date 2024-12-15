import heapq
import operator
import re
import sys
from typing import List, Tuple, Any, Optional

import numpy as np

from aoc.helpers import locate, build_location, read_lines
from aoc.printer import get_meta_from_fn, ANSIColors, print2


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

def even_shape(shape: Tuple[int, int]) -> Tuple[int, int]:
    y = shape[0] if shape[0] % 2 == 0 else shape[0] + 1
    x = shape[1] if shape[1] % 2 == 0 else shape[1] + 1
    return y, x

Candidates = np.ndarray[Any, np.dtype[np.unsignedinteger[np.uint8]]]

def quantize(shape: Tuple[int, int], every_robot: Candidates, z:int, slices: Optional[Tuple[slice, slice]], min_size=3) -> Tuple[int, int, int]:
    y, x = slices if slices else (slice(0, 0), slice(0, 0))
    for dy, dx in ((0, 0), (1, 0), (0, 1), (1, 1)):
        _y = slice(y.start + shape[0] * dy // 2, y.start + shape[0] * (dy + 1) // 2)
        _x = slice(x.start + shape[1] * dx // 2, x.start + shape[1] * (dx + 1) // 2)
        if _y.stop - _y.start < min_size or _x.stop - _x.start < min_size:
            continue
        yield int(every_robot[z, _y, _x].sum()), _y, _x

def every_robot_position_as_3d_array(grid_shape: Tuple[int, int], actual_shape: Tuple[int, int], repeat_at: int, robots: List[List[int]]) -> Candidates:
    """
    Initializes an array a bit greedily at 20_000 at z-index. Alternate approaches:
    # 1. Symbolic math (sympy) for when all robots end up starting position again (k)=
    # 2. GCD
    # 3. hash
    """
    candidates = np.zeros((20_000, *grid_shape), dtype=np.uint8)
    for z in range(1, repeat_at):
        moved_robots = move(actual_shape, robots, z)
        for name, y, x in moved_robots:
            candidates[z, y, x] = 1

    return candidates

def create_priority_queue(repeat_at: int, grid_shape: Tuple[int, int], candidates: Candidates):
    heap = []
    heapq.heapify(heap)
    for z in range(1, repeat_at):
        for sum_, _y, _x in quantize(grid_shape, candidates, z, None):
            potential = (_y.stop - _y.start) * (_x.stop - _x.start)
            heapq.heappush(heap, (float(-sum_) / potential, sum_, z, _y, _x))
    return heap

def solve_(__input=None):
    """
    :expect: 7338
    """
    robots = []
    with open(locate(__input), "r") as fp:
        for line in read_lines(fp):
            robot = list(map(int, re.findall(r"-?\d+", line)))
            robots.append(robot)

    shape = (7, 11) if "test" in __input else (103, 101)
    grid_shape = even_shape(shape)

    repeat_at = 10403
    every_robot_position = every_robot_position_as_3d_array(grid_shape, shape, repeat_at, robots)
    heap = create_priority_queue(repeat_at, grid_shape, every_robot_position)

    while heap:
        _, sum_, z, y, x = heapq.heappop(heap)
        shape = y.stop - y.start, x.stop - x.start

        if sum_ == (y.stop - y.start) * (x.stop - x.start):  # every value in quant has robot
            return z

        for sum_, _y, _x in quantize(shape, every_robot_position, z, (y, x)):
            _shape = _y.stop - _y.start, _x.stop - _x.start
            _potential = operator.mul(*_shape)
            heapq.heappush(heap, (-sum_ / _potential, sum_, z, _y, _x))

if __name__ == "__main__":
    challenge = get_meta_from_fn(solve_, "challenge")
    expect = get_meta_from_fn(solve_, "expect")
    print2(solve_, test_input, challenge)
    print2(solve_, puzzle_input, expect, ANSIColors.OK_GREEN)
