import math
import re
import sys
from math import gcd

import more_itertools
from sympy import symbols, Eq
from sympy.solvers import solve

from aoc.helpers import locate, build_location, read_lines
from aoc.printer import get_meta_from_fn, ANSIColors, print2
from year_2024.day_13.solve_1 import is_integer

sys.setrecursionlimit(30_000)

_default_puzzle_input = "year_2024/day_01/puzzle.txt"
_default_test_input = "year_2024/day_01/test.txt"

puzzle_input = build_location(__file__, "puzzle.txt")
test_input = build_location(__file__, "test.txt")

def _solve(_input=None):
    """
    :challenge: 0
    :expect: 76358113886726
    """
    lines = []
    with open(locate(_input), "r") as fp:
        for line in read_lines(fp):
            lines.append(line)

    machines = []
    for button_a, button_b, prize_ in more_itertools.chunked(lines, 3):
        assert "Button A" in button_a
        assert "Button B" in button_b
        assert "Prize" in prize_
        a = map(int, re.findall(r"\d+", button_a))
        b = map(int, re.findall(r"\d+", button_b))
        prize = map(int, re.findall(r"\d+", prize_))
        machines.append((a, b, prize))

    total = 0
    for n, opt_machine in enumerate(machines):
        _ax, _ay = opt_machine[0]
        _bx, _by = opt_machine[1]
        _px, _py = opt_machine[2]
        a, b, ax, bx, ay, by, px, py = symbols("a b ax bx ay by px py")

        eq0 = Eq(a * ax + b * bx, _px)
        eq1 = Eq(a * ay + b * by, _py)
        eq2 = eq0.subs({ax: _ax, bx: _bx})
        eq3 = eq1.subs({ay: _ay, by: _by})
        solve0 = solve([eq2, eq3], (a, b))
        if is_integer(solve0[a]) and is_integer(solve0[b]):
            total += solve0[a] * 3 + solve0[b]

    return total


if __name__ == "__main__":
    expect = get_meta_from_fn(_solve, "expect")
    print2(_solve, puzzle_input, expect, ANSIColors.OK_GREEN)
