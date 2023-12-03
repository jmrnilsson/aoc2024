import itertools
import operator
import re
import sys
from functools import reduce

from aoc.helpers import locate, build_location, read_lines
from aoc.poll_printer import PollPrinter

# Override only if necessary
_default_puzzle_input = "year_2023/day_01/puzzle.txt"
_default_test_input = "year_2023/day_01/test.txt"

puzzle_input = build_location(__file__, "puzzle.txt")
test_input = build_location(__file__, "test.txt")
test_input_2 = build_location(__file__, "test_2.txt")

challenge_1 = 8
challenge_2 = 2286

def possible_game(rounds, cubes):
    for r, (colour, threshold) in itertools.product(rounds, cubes.items()):
        if r.get(colour) and threshold < r[colour]:
            return False
    return True

def solve_1(input_=None):
    """
    :challenge: 8
    :expect: 2771
    """
    is_test = 1 if "test" in input_ else 0

    sum_of_possible_game_ids = 0
    games = {}

    cubes = {"red": 12, "green": 13, "blue": 14}

    with open(locate(input_), "r") as fp:
        lines = read_lines(fp)

    for line in lines:
        game_str, rounds_str = line.split(":")
        _, game = game_str.split(" ")
        round_rows = rounds_str.split(";")
        rounds = []
        for r in round_rows:
            group_rows = r.split(",")
            groups = {}
            for g in group_rows:
                x = re.findall(r"(\d+)\s+(\w+)", g)[0]
                point_str, colour = x
                point = int(point_str)
                groups[colour] = point
            rounds.append(groups)

        games[game] = rounds

    return sum([int(game) for game, rounds in games.items() if possible_game(rounds, cubes)])

def solve_2(input_=None):
    """
    :challenge: 2286
    :expect: 70924
    """
    _IS_TEST_INPUT_ = True if "test" in input_ else False

    min_draws = {}

    cubes = {"red": 12, "green": 13, "blue": 14}

    with open(locate(input_), "r") as fp:
        lines = read_lines(fp)

    for line in lines:
        game_str, rounds_str = line.split(":")
        _, game = game_str.split(" ")
        draws = []
        for r in rounds_str.split(";"):
            groups_r = r.split(",")
            groups = {}
            for g in groups_r:
                point_str, colour = re.search(r"(\d+)\s+(\w+)", g).groups()
                groups[colour] = int(point_str)
            draws.append(groups)

        min_draws[game] = draws

    max_draws_per_colour_and_game = {}
    for game, draws in min_draws.items():
        min_draws = {"red": 0, "green": 0, "blue": 0}
        for r in draws:
            for colour, threshold in min_draws.items():
                if r.get(colour) and threshold < r[colour]:
                    min_draws[colour] = r[colour]

        max_draws_per_colour_and_game[game] = min_draws

    min_draws = []
    for game, min_colours in max_draws_per_colour_and_game.items():
        if not _IS_TEST_INPUT_:
            min_draws.append(min_colours.values())
        else:
            qualifier_predicate: bool = len(min_colours.keys()) < 3
            for colour, n in cubes.items():
                qualifier_predicate |= min_colours.get(colour) < n

            if qualifier_predicate:
                min_draws.append(min_colours.values())

    return sum([reduce(operator.mul, game, 1) for game in min_draws])


if __name__ == "__main__":
    """
    PWSH
    while ($true) {python -m aoc.year_2023.day_01.solve_1 -poll; start-sleep -seconds 1} 

    BASH
    watch -n 3 python3 -m aoc.year_2023.day_01.day -p
    """
    poll_printer = PollPrinter(solve_1, solve_2, challenge_1, challenge_2, test_input, puzzle_input)
    poll_printer.run(sys.argv)

