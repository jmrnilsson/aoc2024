from typing import List

import pytest

from aoc.printer import get_meta_from_fn
from aoc.tests.test_fixtures import make_fixture, AdventFixture
from . import solve_1 as solution_1
# from .sink import solve_2 as solution_2


class TestAdvent202405:

    def test_part_1_challenge(self):
        assert solution_1.solve_(solution_1.test_input) == get_meta_from_fn(solution_1.solve_, "challenge")

    def test_part_1_expected(self):
        assert solution_1.solve_(solution_1.puzzle_input) == get_meta_from_fn(solution_1.solve_, "expect")

    def _test_part_2_challenge(self, fixt: List[AdventFixture]):
        pass
        # assert solution_2.solve_(solution_2.test_input) == int(fixt[2].challenge)

    def _test_part_2_expected(self, fixt: List[AdventFixture]):
        pass
        # assert solution_2.solve_(solution_2.puzzle_input) == int(fixt[2].expect)
