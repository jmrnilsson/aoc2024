import sys

import numpy as np

from aoc.helpers import locate, build_location, read_lines
from aoc.printer import get_meta_from_fn, print_
from year_2024.day_10.solve_1 import HikingAutomaton

sys.setrecursionlimit(30_000)


_default_puzzle_input = "year_2024/day_01/puzzle.txt"
_default_test_input = "year_2024/day_01/test.txt"

puzzle_input = build_location(__file__, "puzzle.txt")
test_input = build_location(__file__, "test.txt")
test_input_2 = build_location(__file__, "test_2.txt")


def solve(__input=None):
    """
    :challenge: 81
    :expect: 1034
    """
    lines = []
    with open(locate(__input), "r") as fp:
        for line in read_lines(fp):
            a = list(map(int, list(line)))
            lines.append(a)

    grid = np.matrix(lines)
    starting_positions = [[(int(y), int(x))] for y, x in np.argwhere(grid == 0)]

    hiker = HikingAutomaton(starting_positions, grid)
    while not hiker.is_accepting():
        hiker.walk()

    return hiker.rat()


if __name__ == "__main__":
    challenge = get_meta_from_fn(solve, "challenge")
    expect = get_meta_from_fn(solve, "expect")
    print_(solve, test_input, puzzle_input, challenge, expect)
