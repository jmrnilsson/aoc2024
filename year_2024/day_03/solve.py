import re

from aoc.helpers import locate, build_location, read_lines
from aoc.printer import get_meta_from_fn, print_


_default_puzzle_input = "year_2024/day_01/puzzle.txt"
_default_test_input = "year_2024/day_01/test.txt"

puzzle_input = build_location(__file__, "puzzle.txt")
test_input = build_location(__file__, "test.txt")
test_input_2 = build_location(__file__, "test_2.txt")


def solve_1(input_=None):
    """
    :challenge: 161
    :expect: 188741603
    """
    total = 0
    with open(locate(input_), "r") as fp:
        for line in read_lines(fp):
            for a, b in re.findall(r"mul\((\d+),(\d+)\)", line):
                total += int(a) * int(b)

    return total

class MultiplierAutomaton:
    multiply: bool
    total: int

    def __init__(self, start_state: bool):
        self.multiply = start_state
        self.total = 0

    def process(self, commands: re.Match[str]):
        if commands[0] == "do()":
            self.multiply = True
        elif commands[0] == "don't()":
            self.multiply = False
        elif self.multiply:
            self.total += int(commands[1]) * int(commands[2])

def solve_2(input_=None):
    """
    :challenge: 48
    :expect: 67269798
    """
    automaton = MultiplierAutomaton(True)
    with open(locate(input_), "r") as fp:
        for line in read_lines(fp):
            for match in re.finditer(r"mul\((\d+),(\d+)\)|don't\(\)|do\(\)", line):
                automaton.process(match)

    return automaton.total

if __name__ == "__main__":
    challenge = get_meta_from_fn(solve_1, "challenge")
    expect = get_meta_from_fn(solve_1, "expect")
    print_(solve_1, test_input, puzzle_input, challenge, expect)

    challenge = get_meta_from_fn(solve_2, "challenge")
    expect = get_meta_from_fn(solve_2, "expect")
    print_(solve_2, test_input_2, puzzle_input, challenge, expect)
