import re
from collections import Counter
from typing import List

from aoc.helpers import build_location, locate, read_lines
from aoc.printer import get_meta_from_fn, print_


_default_puzzle_input = "year_2024/day_01/puzzle.txt"
_default_test_input = "year_2024/day_01/test.txt"

puzzle_input = build_location(__file__, "puzzle.txt")
test_input = build_location(__file__, "test.txt")
test_input_2 = build_location(__file__, "test_2.txt")


def solve_1(input_=None):
    """
    :challenge: 11
    :expect: 1660292
    """
    left, right = [], []

    with open(locate(input_), "r") as fp:
        for line in read_lines(fp):
            l, r = list(map(int, re.findall(r"\d+", line)))
            left.append(l)
            right.append(r)

    left.sort(), right.sort()
    return sum([abs(left[i] - right[i]) for i, _ in enumerate(left)])


def solve_2(input_=None):
    """
    :challenge: 31
    :expect: 22776016
    """
    left: List[int] = []
    right_counter = Counter()

    with open(locate(input_), "r") as fp:
        for line in read_lines(fp):
            [l, r] = list(map(int, re.findall(r"\d+", line)))
            left.append(l)
            right_counter.update({r: 1})

    return sum(left_item * right_counter[left_item] for left_item in left)


if __name__ == "__main__":
    challenge = get_meta_from_fn(solve_1, "challenge")
    expect = get_meta_from_fn(solve_1, "expect")
    print_(solve_1, test_input, puzzle_input, challenge, expect)

    challenge = get_meta_from_fn(solve_2, "challenge")
    expect = get_meta_from_fn(solve_2, "expect")
    print_(solve_2, test_input, puzzle_input, challenge, expect)
