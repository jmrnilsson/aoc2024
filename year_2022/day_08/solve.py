import operator
import sys
from collections import defaultdict
from functools import reduce
from itertools import takewhile, product

from aoc.helpers import locate, build_location, read_lines
from aoc.poll_printer import PollPrinter
from aoc.tools import transpose

# ICE
_default_puzzle_input = "year_2022/day_01/puzzle.txt"
_default_test_input = "year_2022/day_01/test.txt"

puzzle_input = build_location(__file__, "puzzle.txt")
test_input = build_location(__file__, "test.txt")
test_input_2 = build_location(__file__, "test_2.txt")

challenge_solve_1 = 21
challenge_solve_2 = "exception"


def solve_1(input_=None):
    """
    test=58
    expect=1598415
    """
    with open(locate(input_), "r") as fp:
        matrix = [[int(d) for d in list(line)] for line in read_lines(fp)]

    y_range = [("y", n, axis) for n, axis in enumerate(matrix)]
    x_range = [("x", n, axis) for n, axis in enumerate(transpose(matrix))]
    visible_trees = set()
    for axis, _k, _k_range in y_range + x_range:
        for reverse in (False, True):
            ceiling = -1
            k_range = enumerate(_k_range) if not reverse else reversed(list(enumerate(_k_range)))
            for n, tree in k_range:
                coords = (_k, n) if axis == "y" else (n, _k)
                if tree > ceiling:
                    visible_trees.add(coords)
                ceiling = max(tree, ceiling)

    return sum(1 for _ in visible_trees)


def count_while_less_than_add_one(k_range, value):
    total = sum(1 for _ in takewhile(lambda v: v < value, k_range))
    total += 1 if len(k_range) > total else 0
    return total


def splice_at(k_range, n):
    return k_range[slice(n + 1, len(k_range))], list(reversed(k_range[0: n]))


def solve_2(input_=None):
    """
    test=58
    expect=1598415
    """
    scenic = defaultdict(list)

    with open(locate(input_), "r") as fp:
        matrix = [[int(d) for d in list(line)] for line in read_lines(fp)]

    transposed = transpose(matrix)
    x_len, y_len = len(matrix[0]), len(matrix)
    for y, x in product(range(0, y_len), range(0, x_len)):
        x_range, y_range, value = matrix[y],  transposed[x], matrix[y][x]
        down, up = splice_at(y_range, y)
        right, left = splice_at(x_range, x)
        view = [
            count_while_less_than_add_one(right, value),
            count_while_less_than_add_one(left, value),
            count_while_less_than_add_one(down, value),
            count_while_less_than_add_one(up, value)
        ]
        scenic[(y, x)] = view

    scenic_view = {k: reduce(operator.mul, v,  1) for k, v in scenic.items()}
    return max(scenic_view.values())


if __name__ == "__main__":
    """
    PWSH
    while ($true) {python -m aoc.year_2022.day_03.solve -poll; start-sleep -seconds 1} 

    BASH
    watch -n 3 python3 -m aoc.year_2021.day_03.day -p
    """
    poll_printer = PollPrinter(solve_1, solve_2, challenge_solve_1, challenge_solve_2, test_input, puzzle_input)
    poll_printer.run(sys.argv)

