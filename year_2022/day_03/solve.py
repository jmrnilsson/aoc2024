import re
import re
import sys

from more_itertools import chunked

from aoc.helpers import locate, build_location
from aoc.poll_printer import PollPrinter

# ICE
_default_puzzle_input = "year_2022/day_01/puzzle.txt"
_default_test_input = "year_2022/day_01/test.txt"

puzzle_input = build_location(__file__, "puzzle.txt")
test_input = build_location(__file__, "test.txt")
test_input_2 = build_location(__file__, "test_2.txt")

challenge_solve_1 = 157
challenge_solve_2 = 70


def solve_1(input_=None):
    """
    test=157
    expect=8105
    """
    seed, value = [], 0
    with open(locate(input_), "r") as fp:
        for line in (line for line in fp.readlines() if line != "\n"):
            seed.append([i for i in re.findall(r"\w+", line)][0])

    for rucksacks, length in ((r, len(r) // 2) for r in seed):
        char, = set(rucksacks[:length]) & set(rucksacks[length:])
        value += ord(char) - 96 if char == char.lower() else ord(char) - 38

    return value


def solve_2(input_=None):
    """
    test=70
    expect=2363
    """
    seed, value = [], 0,
    with open(locate(input_), "r") as fp:
        for chunk in chunked((line for line in fp.readlines() if line != "\n"), 3):
            seed.append([i for i in re.findall(r"\w+", line)][0] for line in chunk)

    for rucksacks in seed:
        char, = set.intersection(*[set(r) for r in rucksacks])
        value += ord(char) - 96 if char == char.lower() else ord(char) - 38

    return value


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


