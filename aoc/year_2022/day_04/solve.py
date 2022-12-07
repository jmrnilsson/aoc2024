import re
import sys

from aoc.helpers import locate, build_location, read_lines
from aoc.poll_printer import PollPrinter

# ICE
_default_puzzle_input = "year_2022/day_01/puzzle.txt"
_default_test_input = "year_2022/day_01/test.txt"

puzzle_input = build_location(__file__, "puzzle.txt")
test_input = build_location(__file__, "test.txt")
test_input_2 = build_location(__file__, "test_2.txt")

challenge_solve_1 = 2
challenge_solve_2 = 4


def solve_1(input_=None):
    """
    test=2
    expect=518
    """
    total = 0

    with open(locate(input_), "r") as fp:
        for line in read_lines(fp):
            current, other = [[int(value) for value in i.split("-")] for i in re.findall(r"\d+-\d+", line)]
            if current[0] >= other[0] and current[1] <= other[1] or other[0] >= current[0] and other[1] <= current[1]:
                total += 1

    return total


def solve_2(input_=None):
    """
    test=4
    expect=909
    """
    total = 0

    with open(locate(input_), "r") as fp:
        for line in read_lines(fp):
            both = [(int(i.split("-")[0]), int(i.split("-")[1])) for i in re.findall(r"\d+-\d+", line)]
            current, other = both
            current_min, current_max, other_min, other_max = current[0], current[1], other[0], other[1]

            if set(range(current_min, current_max + 1)) & set(range(other_min, other_max + 1)):
                if current_min <= other_min <= current_max or current_min <= other_max <= current_max:
                    total += 1
                elif other_min <= current_min <= other_max or other_min <= current_max <= other_max:
                    total += 1

    return total


if __name__ == "__main__":
    """
    PWSH
    while ($true) {python -m aoc.year_2022.day_03.solve -poll; start-sleep -seconds 1} 

    BASH
    watch -n 3 python3 -m aoc.year_2021.day_03.day -p
    """
    poll_printer = PollPrinter(solve_1, solve_2, challenge_solve_1, challenge_solve_2, test_input, puzzle_input)
    args = sys.argv[1:]
    if not args:
        poll_printer.print_timed()
    elif re.match("^-poll$|^-p$", args[0]):
        poll_printer.poll_print()
    elif re.match("^-json1$|^-j1$", args[0]):
        poll_printer.poll_json_1()
    elif re.match("^-json2$|^-j2$", args[0]):
        poll_printer.poll_json_2()


