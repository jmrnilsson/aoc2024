import math
import re
import sys
from math import gcd

import more_itertools
from sympy import symbols, Eq
from sympy.solvers import solve

from aoc.helpers import locate, build_location, read_lines
from aoc.printer import get_meta_from_fn, ANSIColors, print2

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

    machines = list()
    for machine in more_itertools.chunked(lines, 3):
        assert "Button A" in machine[0]
        assert "Button B" in machine[1]
        assert "Prize" in machine[2]
        a = list(map(int, re.findall(r"\d+", machine[0])))
        b = list(map(int, re.findall(r"\d+", machine[1])))
        _prize = list(map(int, re.findall(r"\d+", machine[2])))
        prize = [d + 10000000000000 for d in _prize]
        machines.append((a, b, prize))

    optimized_machines = []
    for machine in machines:
        prize_x, prize_y = machine[2]
        a_x, a_y = machine[0]
        b_x, b_y = machine[1]
        gcd_m = gcd(prize_y, a_y, b_y, prize_x, a_x, b_x)
        optimized_machines.append((
            [a_x // gcd_m, a_y // gcd_m],
            [b_x // gcd_m, b_y // gcd_m],
            [prize_x // gcd_m, prize_y // gcd_m]
        ))
        assert not any((a_x % gcd_m, a_y % gcd_m, b_x % gcd_m, b_y % gcd_m, prize_x % gcd_m, prize_y % gcd_m))


    total = 0
    for n, opt_machine in enumerate(optimized_machines):
        _ax, _ay = opt_machine[0]
        _bx, _by = opt_machine[1]
        _px, _py = opt_machine[2]
        a, b, ax, bx, ay, by, px, py = symbols("a b ax bx ay by px py")

        eq0 = Eq(a * ax + b * bx, _px)
        eq1 = Eq(a * ay + b * by, _py)
        eq2 = eq0.subs({ax: _ax, bx: _bx})
        eq3 = eq1.subs({ay: _ay, by: _by})
        solve0 = solve([eq2, eq3], (a, b))
        a_is_int = math.floor(solve0[a]) == solve0[a]
        b_is_int = math.floor(solve0[b]) == solve0[b]
        if a_is_int and b_is_int:
            total += solve0[a] * 3 + solve0[b]

    return total


if __name__ == "__main__":
    challenge = get_meta_from_fn(_solve, "challenge")
    challenge_2 = 772
    challenge_3 = 1930
    challenge_4 = (
        2 * 6 +  # C
        1 * 4 +  # D
        1 * 4 +  # B
        1 * 4 +  # Y
        7 * 15   # A
    )
    challenge_5 = (
        2 * 6 +  # C
        1 * 4 +  # D
        2 * 6 +  # B
        1 * 4 +  # Y
        1 * 4 +  # A1
        4 * 12   # A2
    )

    expect = get_meta_from_fn(_solve, "expect")
    print2(_solve, puzzle_input, expect, ANSIColors.OK_GREEN)
