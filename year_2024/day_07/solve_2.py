import operator
import re
import sys
from functools import reduce
from typing import List, Callable, Tuple

from aoc.helpers import locate, build_location, read_lines
from aoc.printer import print_, get_meta_from_fn

sys.setrecursionlimit(3000)


_default_puzzle_input = "year_2024/day_01/puzzle.txt"
_default_test_input = "year_2024/day_01/test.txt"

puzzle_input = build_location(__file__, "puzzle.txt")
test_input = build_location(__file__, "test.txt")
test_input_2 = build_location(__file__, "test_2.txt")


def concat(left: int, right: int):
    quotient = right
    while quotient := quotient // 10:
        left *= 10

    return left * 10 + right


def computable(acc: Tuple[int, List[int]], value: int):
    answer, seed = acc
    return answer, [v
        for s in seed
        for v in (
            operator.mul(s, value),
            operator.add(s, value),
            concat(s, value)
        )
        if answer >= v
     ]


def calibration_possible(row: List[int]):
    answer, first, *rest = row
    while 1:
        _, remainder = reduce(computable, rest, (answer, [first]))
        if answer in remainder:
            return answer
        elif not remainder:
            return 0
        rest = remainder

def solve(__input=None, prefer_fn: Callable | None  = None):
    """
    :challenge: 11387
    :expect: 169122112716571
    """
    width = 0
    rows: List[List[int]] = []
    with open(locate(__input), "r") as fp:
        for _, line in enumerate(read_lines(fp)):
            items = list(map(int, re.findall(r"\d+", line)))
            rows.append(items)
            width = max(width, len(items))

    return sum(calibration_possible(r) for r in rows)


if __name__ == "__main__":
    challenge = get_meta_from_fn(solve, "challenge")
    expect = get_meta_from_fn(solve, "expect")
    print_(solve, test_input, puzzle_input, challenge, expect)
