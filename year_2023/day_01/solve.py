import itertools
import re
import sys
from functools import reduce
from typing import List, Dict, Tuple

from aoc.helpers import locate, build_location, read_lines
from aoc.poll_printer import PollPrinter

# Override only if necessary
_default_puzzle_input = "year_2023/day_01/puzzle.txt"
_default_test_input = "year_2023/day_01/test.txt"

puzzle_input = build_location(__file__, "puzzle.txt")
test_input = build_location(__file__, "test.txt")
test_input_2 = build_location(__file__, "test_2.txt")

challenge_1 = 142
challenge_2 = 281


def solve_1(input_=None):
    """
    :challenge: 142
    :expect: 53651
    """

    digits = []

    with open(locate(input_), "r") as fp:
        lines = read_lines(fp)

    for line in lines:
        first, last = None, None
        for char in list(line):
            try:
                _ = int(char)
                last = char
                if first is None:
                    first = char
            except ValueError:
                pass

        digits.append(int(first + last))

    return sum(digits)


class DigitParser:
    tokens: Dict[str, int] = {
        "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9
    }

    def to_integer(self, match):
        return self.tokens[match] if self.tokens.get(match) else int(match)

    def parse(self, value: str):
        i, first, j, last = float("inf"), None, -float("inf"), None

        matches: List[Tuple[str, int, int]] = [
            (match.group(), match.end(), self.to_integer(match.group()))
            for pattern in itertools.chain([r"\d"], self.tokens.keys())
            for match in re.finditer(pattern, value)
        ]

        for g, e, value in matches:
            if e < i:
                i, first = e, value
            if e > j:
                j, last = e, value

        return int(f"{first}{last}")

def solve_2(input_=None):
    """
    :challenge: 281
    :expect: 53894
    """
    with open(locate(input_), "r") as fp:
        lines = read_lines(fp)

    digit_parser = DigitParser()
    return reduce(lambda a, b: a + digit_parser.parse(b), lines, 0)


if __name__ == "__main__":
    poll_printer = PollPrinter(solve_1, solve_2, challenge_1, challenge_2, test_input, puzzle_input)
    poll_printer.run(sys.argv)
