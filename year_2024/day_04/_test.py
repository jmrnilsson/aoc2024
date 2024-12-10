from typing import List

import pytest

from aoc.tests.test_fixtures import make_fixture, AdventFixture
from . import solve as solution


class TestAdvent202405:

    @pytest.fixture
    def fixt(self) -> List[AdventFixture]:
        return [None, make_fixture(solution.solve_1), make_fixture(solution.solve_2)]

    def test_part_1_challenge(self, fixt: List[AdventFixture]):
        assert solution.solve_1(solution.test_input) == int(fixt[1].challenge)

    def test_part_1_expected(self, fixt: List[AdventFixture]):
        assert solution.solve_1(solution.puzzle_input) == int(fixt[1].expect)

    def test_part_2_challenge(self, fixt: List[AdventFixture]):
        assert solution.solve_2(solution.test_input) == int(fixt[2].challenge)

    def test_part_2_expected(self, fixt: List[AdventFixture]):
        assert solution.solve_2(solution.puzzle_input) == int(fixt[2].expect)
