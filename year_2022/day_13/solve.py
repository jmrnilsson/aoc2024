import json
import operator
import sys
from enum import Enum
from functools import cmp_to_key
from itertools import zip_longest

from more_itertools import chunked

from aoc.helpers import locate, build_location, read_lines
from aoc.poll_printer import PollPrinter

# ICE
_default_puzzle_input = "year_2022/day_01/puzzle.txt"
_default_test_input = "year_2022/day_01/test.txt"

puzzle_input = build_location(__file__, "puzzle.txt")
test_input = build_location(__file__, "test.txt")
test_input_2 = build_location(__file__, "test_2.txt")

challenge_solve_1 = 13
challenge_solve_2 = "exception"


class Comparison(Enum):
    LT = -1
    EQ = 0
    GT = 1


def ok_pair(left, right):
    if left > right:
        return Comparison.GT
    if left < right:
        return Comparison.LT

    return Comparison.EQ


def ok_all(left_, right_):
    for left, right in zip_longest(left_, right_, fillvalue=-1):
        if isinstance(left, list) and isinstance(right, list):
            if (comparison := ok_all(left, right)) != Comparison.EQ:
                return comparison

        elif isinstance(left, int) and isinstance(right, list):
            if left == -1:
                return Comparison.LT
            if (comparison := ok_all([left], right)) != Comparison.EQ:
                return comparison

        elif isinstance(left, list) and isinstance(right, int):
            if right == -1:
                return Comparison.GT
            if (comparison := ok_all(left, [right])) != Comparison.EQ:
                return comparison

        elif isinstance(left, int) and isinstance(right, int):
            if (comparison := ok_pair(left, right)) != Comparison.EQ:
                return comparison

    return Comparison.EQ


def solve_1(input_=None):
    """
    test=13
    expect=6478
    """
    is_test = 1 if "test" in input_ else 0

    seed = []

    with open(locate(input_), "r") as fp:
        for signal in chunked(read_lines(fp), 2):
            left = json.loads(signal[0])
            right = json.loads(signal[1])
            seed.append((left, right))

    n, correct = 0, []
    for left, right in seed:
        n += 1
        if isinstance(left, int) and isinstance(right, int):
            if ok_pair(left, right) == Comparison.LT:
                correct.append(n)
        elif isinstance(left, list) and isinstance(right, list):
            if ok_all(left, right) == Comparison.LT:
                correct.append(n)
    return sum(correct)


def compare(left, right):
    if comparison := ok_all(left, right) == Comparison.LT:
        return -1
    if comparison == Comparison.GT:
        return 1

    return 0


def solve_2(input_=None):

    """
    test=140
    expect=21922
    """

    seed = []

    with open(locate(input_), "r") as fp:
        for signal in read_lines(fp):
            s = json.loads(signal)
            seed.append(s)

    splicers = [[2]], [[6]]
    seed += list(splicers)
    seed.sort(key=cmp_to_key(compare))
    indices = seed.index(splicers[0]) + 1, seed.index(splicers[1]) + 1

    return operator.mul(*indices)


if __name__ == "__main__":
    """
    PWSH
    while ($true) {python -m aoc.year_2022.day_03.solve -poll; start-sleep -seconds 1} 

    BASH
    watch -n 3 python3 -m aoc.year_2021.day_03.day -p
    """
    poll_printer = PollPrinter(solve_1, solve_2, challenge_solve_1, challenge_solve_2, test_input, puzzle_input)
    poll_printer.run(sys.argv)

