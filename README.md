# 🎅 Advent of Code 2023 🎄

- Install: `requirements install -r requirements.txt`
- Run `python3 year_2023/day_07/solve.py`
- Generate README: `python aoc/template.py`


## year_2023\day_03\solve.py

```py
import itertools
import operator
import sys
from collections import defaultdict
from re import Match, finditer as re_finditer, findall as re_findall
from typing import List, Tuple, Set
from aoc.helpers import locate, build_location, read_lines
from aoc.poll_printer import PollPrinter


def heuristic_distance_from(number_vector: Tuple[int, int, int, int], symbols: Set[Tuple[int, int]]):
    y, xs, xe, number = number_vector
    for x, (symbol_y, symbol_x) in itertools.product(range(xs, xe), symbols):
        if 2 > abs(y - symbol_y) and 2 > abs(x - symbol_x):
            return True
    return False

def solve_1(input_=None):
    """
    :challenge: 4361
    :expect: 535235
    """
    with open(locate(input_), "r") as fp:
        lines = read_lines(fp)

    y_matches: List[List[Tuple[Match, str]]] = [
        [
            (match, pattern)
            for pattern in ("\d+", "\S")
            for match in re_finditer(pattern, line)
            if match.group() != "." and (pattern != "\S" or not re_findall("\d", match.group()))
        ]
        for line in lines
    ]

    symbols: Set[Tuple[int, int]] = set()
    numbers: Set[Tuple[int, int, int, int]] = set()
    for y, y_match in enumerate(y_matches):
        for match, pattern in y_match:
            xs, xe, mg = match.start(), match.end(), match.group()
            if pattern == '\\d+':
                # y, x_start, x_end, number
                numbers.add((y, xs, xe, int(mg)))
            else:
                # y, x
                symbols.add((y, xs))

    return sum(n[-1] for n in numbers if heuristic_distance_from(n, symbols))


def solve_2(input_=None):
    """
    :challenge: 467835
    :expect: 79844424
    """
    with open(locate(input_), "r") as fp:
        lines = read_lines(fp)

    y_matches: List[List[Tuple[Match, str]]] = [
        [
            (match, pattern)
            for pattern in ("\d+", "\*")
            for match in re_finditer(pattern, line)
        ]
        for line in lines
    ]

    symbols: Set[Tuple[int, int]] = set()
    numbers: Set[Tuple[int, int, int, int]] = set()
    for y, y_match in enumerate(y_matches):
        for match, pattern in y_match:
            if pattern == '\\d+':
                # y, x_start, x_end, number
                numbers.add((y, match.start(), match.end(), int(match.group())))
            else:
                symbols.add((y, match.start()))

    gears = defaultdict(set)
    for x_vectors in numbers:
        y, xs, xe, number = x_vectors
        for x, symbol in itertools.product(range(xs, xe), symbols):
            sy, sx = symbol
            if 2 > abs(y - sy) and 2 > abs(x - sx):
                gears[symbol].add(x_vectors)

    return sum(operator.mul(*map(lambda n: n[-1], numbers)) for numbers in gears.values() if len(numbers) == 2)


```
## year_2023\day_02\solve.py

```py
import itertools
import operator
import re
import sys
from functools import reduce
from aoc.helpers import locate, build_location, read_lines
from aoc.poll_printer import PollPrinter


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


```
## year_2023\day_01\solve.py

```py
import itertools
import re
import sys
from functools import reduce
from typing import List, Dict, Tuple
from aoc.helpers import locate, build_location, read_lines
from aoc.poll_printer import PollPrinter


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


```
