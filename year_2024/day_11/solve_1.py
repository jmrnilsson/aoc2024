import functools
import sys
from typing import List

from aoc.helpers import locate, build_location, read_lines
from aoc.printer import get_meta_from_fn, print_


sys.setrecursionlimit(30_000)

_default_puzzle_input = "year_2024/day_01/puzzle.txt"
_default_test_input = "year_2024/day_01/test.txt"

puzzle_input = build_location(__file__, "puzzle.txt")
test_input = build_location(__file__, "test.txt")
test_input_2 = build_location(__file__, "test_2.txt")

def int_split(number: int):
    original_number = number
    power_of = 0
    quotient = number
    while quotient := quotient // 10:
        power_of += power_of

    top = original_number // power_of // 2 * 10
    bottom = original_number - top
    return top, bottom

@functools.cache
def split(number: int):
    strn = str(number)
    if (size := len(strn)) % 2 == 0:
        return int(strn[:size//2]), int(strn[size//2:])
    return number,


class ObserverAutomaton:
    stones: List[int]
    count: int
    exit_count: int

    def __init__(self, stones: List[int], exit_count: int):
        self.stones = list(stones)
        self.count = 0
        self.exit_count = exit_count

    def blink(self):
        for i in range(len(self.stones) - 1, -1, -1):
            stone = self.stones[i]
            if stone == 0:
                self.stones[i] = 1
            elif len((maybe_two := split(stone))) > 1:
                self.stones[i] = maybe_two[0]
                self.stones.insert(i + 1, maybe_two[1])
            else:
                self.stones[i] = self.stones[i] * 2024

        self.count += 1

    def is_accepting(self):

        return self.exit_count == self.count

    def number_of_stones(self):
        return len(self.stones)



def solve(__input=None):
    """
    :challenge: 55312
    :expect: 233875
    """
    lines = []
    with open(locate(__input), "r") as fp:
        for line in read_lines(fp):
            stones = [int(d) for d in line.split(" ")]
            lines.append(stones)

    observer = ObserverAutomaton(stones, 25)  # 22
    while not observer.is_accepting():
        observer.blink()

    return observer.number_of_stones()


if __name__ == "__main__":
    challenge = get_meta_from_fn(solve, "challenge")
    expect = get_meta_from_fn(solve, "expect")
    print_(solve, test_input_2, puzzle_input, challenge, expect)
