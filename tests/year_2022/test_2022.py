import importlib
import time
import unittest

import pytest

import aoc.year_2022.day_01.solve as solve
import aoc.year_2022.day_02.solve as solve_2
from aoc.helpers import locate, build_location_test
import importlib
import time
import unittest

import pytest

import aoc.year_2022.day_01.solve as solve
import aoc.year_2022.day_02.solve as solve_2
from aoc.helpers import locate, build_location_test

solutions = [
    [24000, 69281, 45000, 201524],
    [15, 13565, 12, 12424],
    [157, 8105, 70, 2363],
    [2, 518, 4, 909],
    ["CMZ", "VJSFHWGFT", "MCD", "LCTQFBVZV"],
    [7, 1198, 19, 3120],
    [95437, 1350966, 24933642, 6296435]
]


def generate_tests():
    n_day = 0
    for solution in solutions:
        n_day += 1
        expected, n = solution, 0
        for func_name in ["solve_1", "solve_2"]:
            for test_input in ["test", "puzzle"]:
                yield func_name, test_input, expected[n], f"day_{n_day}", n_day
                n += 1


_suite = list(generate_tests())
test_data = [(t[0], t[1], t[2], t[4]) for t in _suite]
ids = [f"{t[3]}_{t[0]}_{t[1]}" for t in _suite]


@pytest.mark.parametrize("func_name,test_input,expected,day", test_data, ids=ids)
def test_day(func_name, test_input, day, expected):
    mod = importlib.import_module(f"aoc.year_2022.day_{str(day).zfill(2)}.solve")
    func = getattr(mod, func_name)
    file_name = "test.txt" if "test" == test_input else "puzzle.txt"
    input_file = build_location_test(2022, day, file_name)

    # ts = datetime.datetime.utcnow()
    ts = time.time_ns()
    actual = func(input_file)
    te = time.time_ns()
    # te = datetime.datetime.utcnow()
    diff = (te - ts) / 1e6
    print(f"test: {day}: {round(diff, 3)}ms")
    assert actual == expected


class TestInputs(unittest.TestCase):
    def test_day_01_puzzle_input_defaults(self):
        default_override = locate(solve._default_puzzle_input)
        default = locate(solve.puzzle_input)
        self.assertEqual(default_override, default)

    def test_day_01_test_input_defaults(self):
        default_override = locate(solve._default_test_input)
        default = locate(solve.test_input)
        self.assertEqual(default_override, default)

    def test_day_02_puzzle_input_defaults(self):
        default_override = locate(solve_2._default_puzzle_input)
        default = locate(build_location_test(2022, 2, "puzzle.txt"))
        self.assertEqual(default_override, default)

    def test_day_02_test_input_defaults(self):
        default_override = locate(solve_2._default_test_input)
        default = locate(build_location_test(2022, 2, "test.txt"))
        self.assertEqual(default_override, default)
