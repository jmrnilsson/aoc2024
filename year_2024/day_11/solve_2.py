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
def quick_split_if_even(stone: int) -> List[int]:
    log_10, quotient = 1, stone
    while quotient := quotient // 10:
        log_10 += 1

    if log_10 % 2 == 1:
        return []

    quotient = 10 ** (log_10 // 2)
    top = stone // quotient
    top_floor = top * quotient
    bottom = stone - top_floor
    return [top, bottom]


class ObserverAutomaton:
    stones: Counter
    iteration: int
    end_on: int

    def __init__(self, stones: List[int], end_on: int):
        self.stones = Counter({s: 1 for s in stones})
        self.iteration = 0
        self.end_on = end_on

    def blink(self):
        materialized = dict(self.stones)
        self.stones.clear()
        for stone, amount in materialized.items():
            if stone == 0:
                self.stones.update({1: amount})
            elif len((maybe_two := quick_split_if_even(stone))) > 1:
                for new_stone in maybe_two:
                    self.stones.update({new_stone: amount})
            else:
                self.stones.update({stone * 2024: amount})

        self.iteration += 1

    def is_accepting(self):
        return self.end_on == self.iteration

    def sum_stones(self):
        return sum(v for _, v in self.stones.items())


def solve(_input=None):
    """
    :challenge: 55312
    :expect: 277444936413293
    """
    lines = []
    with open(locate(_input), "r") as fp:
        for line in read_lines(fp):
            stones = [int(d) for d in line.split(" ")]
            lines.append(stones)

    n: int = 25 if "test" in _input else 75
    observer = ObserverAutomaton(stones, n)  # 22
    while not observer.is_accepting():
        observer.blink()

    return observer.sum_stones()


if __name__ == "__main__":
    challenge = get_meta_from_fn(solve, "challenge")
    expect = get_meta_from_fn(solve, "expect")
    print_(solve, test_input_2, puzzle_input, challenge, expect)
