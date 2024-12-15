from typing import List

import pytest

from aoc.printer import get_meta_from_fn
from aoc.tests.test_fixtures import make_fixture, AdventFixture, get_challenges_from_meta
from . import solve_1 as solution_1
from . import solve_2 as solution_2


class TestAdvent202405:

    @pytest.fixture
    def fixt(self) -> List[AdventFixture]:
        return [None, make_fixture(solution_1.solve_), make_fixture(solution_2.solve_)]

    def test_part_1_challenge(self, fixt: List[AdventFixture]):
        assert solution_1.solve_(solution_1.test_input) == int(fixt[1].challenge)

    def test_part_1_challenge_2(self, fixt: List[AdventFixture]):
        challenge_2 = get_meta_from_fn(solution_1.solve_, "challenge_2")
        assert solution_1.solve_(solution_1.test_input_2) == int(challenge_2)

    def test_part_1_expected(self, fixt: List[AdventFixture]):
        assert solution_1.solve_(solution_1.puzzle_input) == int(fixt[1].expect)

    def test_part_2_challenge(self, fixt: List[AdventFixture]):
        assert solution_2.solve_(solution_2.test_input_2) == int(fixt[2].challenge)

    def test_part_2_expected(self, fixt: List[AdventFixture]):
        assert solution_2.solve_(solution_2.puzzle_input) == int(fixt[2].expect)
