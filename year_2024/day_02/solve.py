import re
from typing import Generator, List, Literal, Set, Tuple

from more_itertools.recipes import sliding_window

from aoc.helpers import build_location, locate, read_lines
from aoc.printer import get_meta_from_fn, print_


_default_puzzle_input = "year_2024/day_01/puzzle.txt"
_default_test_input = "year_2024/day_01/test.txt"

puzzle_input = build_location(__file__, "puzzle.txt")
test_input = build_location(__file__, "test.txt")
test_input_2 = build_location(__file__, "test_2.txt")

def safe(number: Tuple[int] | List[int]):
    ops: Set[Literal["gt", "lt"]] = set()
    for a, b in sliding_window(number, 2):
        if not 0 < abs(a - b) < 4:
            return False
        if a > b:
            ops.add("gt")
        if a < b:
            ops.add("lt")

    return len(ops) == 1


def solve_1(input_=None):
    """
    :challenge: 2
    :expect: 252
    """
    levels = list()
    with open(locate(input_), "r") as fp:
        for line in read_lines(fp):
            levels.append(list(map(int, re.findall(r"\d+", line))))

    return sum(1 for l in levels if safe(l))

def omissions(levels: Tuple[int]) -> Generator[Tuple[int], None, None]:
    for i, _ in enumerate(levels):
        materialized_levels = list(levels)
        materialized_levels.pop(i)
        yield tuple(materialized_levels)

def safe_with_tolerations(level: Tuple[int]):
    return safe(level) or any(1 for o in omissions(level) if safe(o))


def solve_2(input_=None):
    """
    :challenge: 4
    :expect: 324
    """
    levels = list()
    with open(locate(input_), "r") as fp:
        for line in read_lines(fp):
            levels.append(tuple(map(int, re.findall(r"\d+", line))))

    return sum(1 for l in levels if safe_with_tolerations(l))

if __name__ == "__main__":
    challenge = get_meta_from_fn(solve_1, "challenge")
    expect = get_meta_from_fn(solve_1, "expect")
    print_(solve_1, test_input, puzzle_input, challenge, expect)

    challenge = get_meta_from_fn(solve_2, "challenge")
    expect = get_meta_from_fn(solve_2, "expect")
    print_(solve_2, test_input, puzzle_input, challenge, expect)
