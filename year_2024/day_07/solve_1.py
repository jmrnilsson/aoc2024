import operator
import re
import sys
from typing import List

from aoc.helpers import locate, build_location, read_lines
from aoc.printer import print_, get_meta_from_fn

sys.setrecursionlimit(3000)


_default_puzzle_input = "year_2024/day_01/puzzle.txt"
_default_test_input = "year_2024/day_01/test.txt"

puzzle_input = build_location(__file__, "puzzle.txt")
test_input = build_location(__file__, "test.txt")
test_input_2 = build_location(__file__, "test_2.txt")


class WalkAutomaton:
    expect: int
    state: List[int]

    def __init__(self, expect_: int, start: int):
        self.expect = expect_
        self.state = [start]

    def eval(self, value: int):
        ops = [(operator.mul, "*"), (operator.add, "+")]
        previous_values = list(self.state)
        values = []
        for previous_value in previous_values:
            for op, _ in ops:
                values.append(op(previous_value, value))
        self.state = values

    def is_accepting(self):
        for v in self.state:
            if v == self.expect:
                return True

        return False


def solve(__input=None):
    """
    :challenge: 3749
    :expect: 1545311493300
    """
    lines: List[List[int]] = []
    with open(locate(__input), "r") as fp:
        for _, line in enumerate(read_lines(fp)):
            items = list(map(int, re.findall(r"\d+", line)))
            lines.append(items)

    total = 0
    for row in lines:
        a = WalkAutomaton(row[0], row[1])
        for value in row[2:]:
            a.eval(value)

        if a.is_accepting():
            total += a.expect

    return total


if __name__ == "__main__":
    challenge = get_meta_from_fn(solve, "challenge")
    expect = get_meta_from_fn(solve, "expect")
    print_(solve, test_input, puzzle_input, challenge, expect)
