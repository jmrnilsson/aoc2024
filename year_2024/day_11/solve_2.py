import functools
import sys
from collections import Counter
from typing import List, Tuple

from aoc.helpers import locate, build_location, read_lines
from aoc.printer import get_meta_from_fn, print_


sys.setrecursionlimit(30_000)

_default_puzzle_input = "year_2024/day_01/puzzle.txt"
_default_test_input = "year_2024/day_01/test.txt"

puzzle_input = build_location(__file__, "puzzle.txt")
test_input = build_location(__file__, "test.txt")
test_input_2 = build_location(__file__, "test_2.txt")

@functools.cache
def quick_log_10(stone: int) -> int:
    log_10, quotient = 1, stone
    while quotient := quotient // 10:
        log_10 += 1

    return log_10

@functools.cache
def split(log_10: int, stone: int) -> Tuple[int, int]:
    quotient = 10 ** (log_10 // 2)
    top = stone // quotient
    top_floor = top * quotient
    bottom = stone - top_floor
    return top, bottom


def observe_stones(_stones: List[int], exit_after: int):
    stones: Counter = Counter()
    iteration: int = 0

    for stone in _stones:
        stones.update({ stone: 1})

    while exit_after != iteration:
        materialized = dict(stones)
        stones.clear()
        for stone, amount in materialized.items():
            if stone == 0:
                stones.update({1: amount})
                continue

            if (log_10 := quick_log_10(stone)) % 2 == 0:
                top, bottom = split(log_10, stone)
                for v in (top, bottom):
                    stones.update({v: amount})
                continue

            stones.update({ stone * 2024: amount})

        iteration += 1

    return sum(v for v in stones.values())


def solve(_input=None):
    """
    :challenge: 55312
    :expect: 277444936413293
    """
    stones = []
    with open(locate(_input), "r") as fp:
        for line in read_lines(fp):
            stones += [int(d) for d in line.split(" ")]

    n: int = 25 if "test" in _input else 75
    no_of_stones = observe_stones(stones, n)
    return no_of_stones



if __name__ == "__main__":
    challenge = get_meta_from_fn(solve, "challenge")
    expect = get_meta_from_fn(solve, "expect")
    print_(solve, test_input_2, puzzle_input, challenge, expect)
