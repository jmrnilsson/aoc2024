import sys
from enum import Enum
from typing import List, Tuple, Set

import numpy as np

from aoc.helpers import locate, build_location, read_lines
from aoc.printer import print_, get_meta_from_fn


sys.setrecursionlimit(3000)


_default_puzzle_input = "year_2024/day_01/puzzle.txt"
_default_test_input = "year_2024/day_01/test.txt"

puzzle_input = build_location(__file__, "puzzle.txt")
test_input = build_location(__file__, "test.txt")
test_input_2 = build_location(__file__, "test_2.txt")

class Accept(Enum):
    Pending = 0
    Stuck = 1
    Outside = 2

class WalkAutomaton:
    pos: Tuple[int, int]
    dir: int
    shape_set = Set[int]
    seen = Set[Tuple[int, int, int]] # grid y, x, dir
    steps = int
    outside: bool
    obstacle: Tuple[int, int]

    def __init__(self, starting_position: Tuple[int, int], starting_direction: int, grid, obstacle: Tuple[int, int]):
        self.pos = starting_position
        self.dir = starting_direction
        self.grid = grid
        self.shape_set = set(grid.shape)
        self.seen = set()
        self.steps = 0
        self.outside = False
        self.obstacle = obstacle

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

        if self.grid[new_pos] == 1 or new_pos == self.obstacle:  # obstructed
            self._turn()
        else:
            self.pos = self._step()
            self.grid[self.pos] = 3

    def is_accepting(self) -> Accept:
        if self.outside:
            return Accept.Outside

        hashable = tuple([*self.pos, self.dir])
        if hashable in self.seen:
            return Accept.Stuck

        self.seen.add(hashable)
        return Accept.Pending


def to_int(text: str):
    match text:
        case "#": return 1
        case "^": return 2
        case _: return 0

def solve(__input=None):
    """
    :challenge: 6
    :expect: 1482
    :notes: A clever solution required or a better brute. Ideas include memoization, warp, traces or GPUs.
    """
    _grid: List[List[int]] = []
    with open(locate(__input), "r") as fp:
        for line in read_lines(fp):
            _grid.append([to_int(character) for character in list(line)])

    grid = np.matrix(_grid, dtype=np.int8)
    starting_position = tuple(np.argwhere(grid == 2)[0])
    candidate_obstruction_positions = [tuple(o) for o in np.argwhere(grid == 0)]
    grid[starting_position] = 3

    total = 0
    for i, obstacle in enumerate(candidate_obstruction_positions):
        if i % 75 == 0:
            print(f"{i} of {len(candidate_obstruction_positions)}")

        walk = WalkAutomaton(starting_position, 0, grid, obstacle)
        while 1:
            walk.walk()
            if accept := walk.is_accepting():
                if accept == Accept.Stuck:
                    total += 1
                    break
                if accept == Accept.Outside:
                    break

    return total


if __name__ == "__main__":
    challenge = get_meta_from_fn(solve, "challenge")
    answer = get_meta_from_fn(solve, "expect")
    print_(solve, test_input, puzzle_input, challenge, answer)
