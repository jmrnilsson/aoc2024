import sys
from typing import List, Tuple, Set

import numpy as np

from aoc.helpers import locate, build_location, read_lines
from aoc.printer import get_meta_from_fn, print_


sys.setrecursionlimit(3000)


_default_puzzle_input = "year_2024/day_01/puzzle.txt"
_default_test_input = "year_2024/day_01/test.txt"

puzzle_input = build_location(__file__, "puzzle.txt")
test_input = build_location(__file__, "test.txt")
test_input_2 = build_location(__file__, "test_2.txt")


class WalkAutomaton:
    pos: Tuple[int, int]
    dir: int
    shape_set = Set[int]
    seen = Set[Tuple[int, int, int]] # grid y, x, dir
    steps = int
    outside: bool

    def __init__(self, starting_pos: Tuple[int, int], starting_direction: int, grid):
        self.pos = starting_pos
        self.dir = starting_direction
        self.grid = grid
        self.shape_set = set(grid.shape)
        self.seen = set()
        self.steps = 0
        self.outside = False

    def _step(self) -> Tuple[int, int]:
        y, x = self.pos
        match self.dir:
            case 0: return y - 1, x + 0
            case 90: return y + 0, x + 1
            case 180: return y + 1, x + 0
            case 270: return y + 0, x + -1

        raise TypeError("What's going on here!")

    def _turn(self):
        new_dir = (90 + self.dir) % 360
        self.dir = new_dir

    def walk(self):
        self.steps += 1
        new_pos = self._step()
        if outside := -1 in new_pos or self.shape_set.intersection(set(new_pos)):
            self.outside = outside
            return

        if self.grid[new_pos] == 1:  # obstructed
            self._turn()
        else:
            self.pos = self._step()
            self.grid[self.pos] = 3

    def is_accepting(self):
        return self.outside


def to(v: str):
    if v == "#":
        return 1
    elif v == "^":
        return 2
    else:
        return 0


def solve_1(__input=None):
    """
    :challenge: 41
    :expect: 4973
    """
    _grid: List[List[int]] = []
    with open(locate(__input), "r") as fp:
        for ind, line in enumerate(read_lines(fp)):
            l = [to(l) for l in list(line)]
            _grid.append(l)

    grid = np.matrix(_grid, dtype=np.int8)
    starting_position = tuple(np.argwhere(grid == 2)[0])
    grid[starting_position] = 3

    walk = WalkAutomaton(starting_position, 0, grid)
    while 1:
        walk.walk()
        if walk.is_accepting():
            break

    return sum(1 for _ in np.argwhere(grid == 3))  # 1 + maybe for pos


if __name__ == "__main__":
    challenge = get_meta_from_fn(solve_1, "challenge")
    expect = get_meta_from_fn(solve_1, "expect")
    print_(solve_1, test_input, puzzle_input, challenge, expect)

