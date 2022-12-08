import re
import sys

from aoc.helpers import locate, build_location, read_lines
from aoc.poll_printer import PollPrinter

# ICE
_default_puzzle_input = "year_2022/day_02/puzzle.txt"
_default_test_input = "year_2022/day_02/test.txt"

puzzle_input = build_location(__file__, "puzzle.txt")
test_input = build_location(__file__, "test.txt")
test_input_2 = build_location(__file__, "test_2.txt")

challenge_solve_1 = 15
challenge_solve_2 = 12


def solve_1(input_=None):
    feed = []

    lookup = {
        "A": "ROCK",
        "X": "ROCK",
        "B": "PAPER",
        "Y": "PAPER",
        "Z": "SCISSORS",
        "C": "SCISSORS",
    }

    with open(locate(input_), "r") as fp:
        for line in read_lines(fp):
            other, me = line.split(" ")
            me_ = me.replace("\n", "")
            feed.append((lookup[other], lookup[me_]))

    point = 0
    for other, me in feed:
        if other == "ROCK" and me == "PAPER":
            point += 6
        if other == "PAPER" and me == "SCISSORS":
            point += 6
        if other == "SCISSORS" and me == "ROCK":
            point += 6

        if other == me:
            point += 3

        if me == "ROCK":
            point += 1
        if me == "PAPER":
            point += 2
        if me == "SCISSORS":
            point += 3

    return point


def solve_2(input_=None):
    feed = []
    lookup = {
        "A": "ROCK",
        "X": "ROCK",
        "B": "PAPER",
        "Y": "PAPER",
        "Z": "SCISSORS",
        "C": "SCISSORS",
    }

    with open(locate(input_), "r") as fp:
        for line in read_lines(fp):
            other, me = line.split(" ")
            me_ = me.replace("\n", "")
            me__ = None
            other_ = lookup[other]
            if me_ == "X":
                if other_ == "ROCK":
                    me__ = "SCISSORS"
                if other_ == "SCISSORS":
                    me__ = "PAPER"
                if other_ == "PAPER":
                    me__ = "ROCK"
            elif me_ == "Y":
                me__ = other_
            else:
                if other_ == "ROCK":
                    me__ = "PAPER"
                if other_ == "SCISSORS":
                    me__ = "ROCK"
                if other_ == "PAPER":
                    me__ = "SCISSORS"
            feed.append((other_, me__))

    point = 0
    for other, me in feed:
        if other == "ROCK" and me == "PAPER":
            point += 6
        if other == "PAPER" and me == "SCISSORS":
            point += 6
        if other == "SCISSORS" and me == "ROCK":
            point += 6

        if other == me:
            point += 3

        if me == "ROCK":
            point += 1
        if me == "PAPER":
            point += 2
        if me == "SCISSORS":
            point += 3

    return point


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


