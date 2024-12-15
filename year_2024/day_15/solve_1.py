import re
import sys
from typing import List, Tuple, Set

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


class StorageRobotAutomaton:

    def __init__(self, grid):
        self.grid = grid

    def assign(self, ys: slice, xs: slice, array, __dir: str):
        i = 0
        for y in range(ys.start, ys.stop):
            for x in range(xs.start, xs.stop):
                self.grid[y, x] = array[i]
                i += 1

    def shove(self, dir_: str, ) -> bool:
        robot, = np.argwhere(self.grid == "@")
        y, x = robot
        reverse = False
        accumulated = ["@"]
        found = False
        pos = [y, x]
        while not found and - 1 < pos[0] < self.grid.shape[0] and - 1 < pos[1] < self.grid.shape[1]:
            match dir_:
                case '^':
                    pos[0] -= 1
                    reverse = True
                case 'v':
                    pos[0] += 1
                case '<':
                    pos[1] -= 1
                    reverse = True
                case '>':
                    pos[1] += 1
                case _:
                    raise NotImplemented("What!")

            value = self.grid[*pos]
            if value == ".":  #  and accumulated:
                y0 = sorted([pos[0], y])
                y0[-1] = y0[-1] + 1
                x0 = sorted([pos[1], x])
                x0[-1] = x0[-1] + 1
                accumulated.insert(0, ".")
                if reverse:
                    accumulated.reverse()

                self.assign(slice(*y0), slice(*x0), accumulated, dir_)
                return True
            elif value == "#":
                return False
            else:
                accumulated.append(value)

    def sum_goods_positioning_system(self):
        blocks = np.argwhere(self.grid == "O")
        return sum(100 * y + x for y, x in blocks)


def solve_(__input=None):
    """
    :challenge: 2028
    :challenge_2: 10092
    :expect: 1526018
    """
    _programming = ""
    _maze = []
    with open(locate(__input), "r") as fp:
        for line in read_lines(fp):
            if re.search(r'[<v^>]', line):
                _programming += line
            else:
                _maze.append(list(line))

    programming = list(_programming)
    grid = np.matrix(_maze, dtype=str)

    storage_robot = StorageRobotAutomaton(grid)
    for direction in programming:
        storage_robot.shove(direction)

    return storage_robot.sum_goods_positioning_system()


if __name__ == "__main__":
    challenge = get_meta_from_fn(solve_, "challenge")
    challenge_2 = 10092
    expect = get_meta_from_fn(solve_, "expect")
    print2(solve_, test_input, challenge)
    print2(solve_, test_input_2, challenge_2, ANSIColors.FAIL)
    print2(solve_, puzzle_input, expect, ANSIColors.OK_GREEN)
