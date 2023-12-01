# 🎄 Advent of Code 2023 🎄

- Install: `requirements install -r requirements.txt`
- Run `python3 year_2023/day_07/solve.py`
- Test: Use IDE :)
- Generate README: `python aoc/template.py`

## year_2023\day_01\solve.py

```py
import itertools
import json
import operator
import re
import sys
from collections import Counter, OrderedDict, defaultdict
from typing import Tuple, Match, List, Dict
import more_itertools
import numpy as np
from defaultlist import defaultlist
from aoc import tools
from aoc.helpers import timing, locate, Printer, build_location, test_nocolor, puzzle_nocolor, read_lines
from aoc.poll_printer import PollPrinter


def solve_1(input_=None):
    """
    test=142
    expect=53651
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

        items: List[Match[str]] = [
            match for pattern in itertools.chain([r"\d"], self.tokens.keys())
            for match in re.finditer(pattern, value)
        ]

        for match in items:
            group, end, value = match.group(), match.end(), self.to_integer(match.group())

            if value:
                if end < i:
                    i, first = end, value
                if end > j:
                    j, last = end, value

        return first, last


def solve_2(input_=None):
    """
    test=281
    expect=53894
    """
    digits = []

    with open(locate(input_), "r") as fp:
        lines = read_lines(fp)

    digit_parser = DigitParser()
    assert digit_parser.parse("ddgjgcrssevensix37twooneightgt") == (7, 8)

    for line in lines:
        min_digit, max_digit = digit_parser.parse(line)
        digits.append(int(f"{min_digit}{max_digit}"))

    return sum(d for d in digits)


```

