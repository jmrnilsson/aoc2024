import sys
from copy import deepcopy
from typing import List, Tuple, Generator

import numpy as np

from aoc.helpers import locate, build_location, read_lines
from aoc.printer import get_meta_from_fn, print_

sys.setrecursionlimit(30_000)


_default_puzzle_input = "year_2024/day_01/puzzle.txt"
_default_test_input = "year_2024/day_01/test.txt"

puzzle_input = build_location(__file__, "puzzle.txt")
test_input = build_location(__file__, "test.txt")
test_input_2 = build_location(__file__, "test_2.txt")


class HikingAutomaton:
    trails: List[List[Tuple[int, int]]]
    done: List[List[Tuple[int, int]]]

    def __init__(self, starting_pos: List[List[Tuple[int, int]]], grid):
        self.trails = list(starting_pos)
        self.done = []
        self.grid = grid

    def in_bound(self, y: int, x: int) -> bool:
        return -1 < y < self.grid.shape[0] and -1 < x < self.grid.shape[1]

    def _step(self, y, x) -> Generator[Tuple[int, int, int], None, None]:
        steps = (
            (y - 1, x + 0),
            (y + 0, x + 1),
            (y + 1, x + 0),
            (y + 0, x + -1)
        )
        for step in steps:
            if self.in_bound(*step) and (value := self.grid[step]) == self.grid[y, x] + 1:
                yield *step, value

    def walk(self):
        trails = list(self.trails)
        self.trails.clear()
        for n, trail_ in enumerate(trails):
            last_step = trail_[-1]
            for y, x, value in self._step(*last_step):
                trail = deepcopy(trail_)
                trail.append((y, x))
                if value == 9:
                    self.done.append(trail)
                else:
                    self.trails.append(trail)

    def score(self):
        head_and_tails = {
            (trail[0], trail[-1])
            for trail in self.done
        }
        return len(head_and_tails)

    def rat(self):
        return len(self.done)

    def is_accepting(self):
        return len(self.trails) == 0


def solve(__input=None):
    """
    :challenge: 36
    :expect: 459
    """
    lines = []
    with open(locate(__input), "r") as fp:
        for line in read_lines(fp):
            lines.append(list(map(int, list(line))))

    grid = np.matrix(lines)
    starting_positions = [[(int(y), int(x))] for y, x in np.argwhere(grid == 0)]

    hiker = HikingAutomaton(starting_positions, grid)
    while not hiker.is_accepting():
        hiker.walk()

    return hiker.score()


if __name__ == "__main__":
    challenge = get_meta_from_fn(solve, "challenge")
    expect = get_meta_from_fn(solve, "expect")
    print_(solve, test_input, puzzle_input, challenge, expect)
