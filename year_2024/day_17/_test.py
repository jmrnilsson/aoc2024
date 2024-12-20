from typing import List

import more_itertools
import pytest

from aoc.tests.test_fixtures import make_fixture, AdventFixture
from . import solve_1 as solution_1
# from . import solve_2 as solution_2
from .solve_1 import ChronospatialComputer


class TestAdvent202405:

    def test_part_1_challenge(self):
        assert solution_1.solve_(solution_1.test_input) == "4,6,3,5,6,3,5,2,1,0"

    def test_part_1_expected(self):
        assert solution_1.solve_(solution_1.puzzle_input) == '2,1,3,0,5,2,3,7,1'

    def _test_part_2_challenge(self, fixt: List[AdventFixture]):
        pass
        # assert solution_2.solve_(solution_2.test_input) == int(fixt[2].challenge)

    def _test_part_2_expected(self, fixt: List[AdventFixture]):
        pass
        # assert solution_2.solve_(solution_2.puzzle_input) == int(fixt[2].expect)

    def test_example_custom_1(self):
        computer = ChronospatialComputer(123, 1, 1, [0, 0])
        while optcode := computer.next():
            computer.set_operation(optcode[0])
            computer.set_operand(optcode[1])
            computer.apply()

        assert computer.a.get() == 123

    def test_example_custom_2(self):
        computer = ChronospatialComputer(8, 1, 1, [0, 1])
        while optcode := computer.next():
            computer.set_operation(optcode[0])
            computer.set_operand(optcode[1])
            computer.apply()

        assert computer.a.get() == 4


    def test_example_custom_bxl(self):
        computer = ChronospatialComputer(-1, 3, -1, [1, 1])
        while optcode := computer.next():
            computer.set_operation(optcode[0])
            computer.set_operand(optcode[1])
            computer.apply()

        assert computer.b.get() == 2

    def test_example_custom_bxc(self):
        computer = ChronospatialComputer(-1, 6, 2, [4, 20])
        while optcode := computer.next():
            computer.set_operation(optcode[0])
            computer.set_operand(optcode[1])
            computer.apply()

        assert computer.b.get() == 4

    def test_example_custom_out_a(self):
        actual = []
        computer = ChronospatialComputer(10, 12, 14, [5, 4], actual.append)
        while optcode := computer.next():
            computer.set_operation(optcode[0])
            computer.set_operand(optcode[1])
            computer.apply()

        assert "".join(actual) == '2,'

    def test_example_custom_out_b(self):
        actual = []
        computer = ChronospatialComputer(10, 12, 14, [5, 5], actual.append)
        while optcode := computer.next():
            computer.set_operation(optcode[0])
            computer.set_operand(optcode[1])
            computer.apply()

        assert "".join(actual) == '4,'

    def test_example_custom_out_c(self):
        actual = []
        computer = ChronospatialComputer(10, 12, 13, [5, 6], actual.append)
        while optcode := computer.next():
            computer.set_operation(optcode[0])
            computer.set_operand(optcode[1])
            computer.apply()

        assert "".join(actual) == '5,'

    def test_example_custom_bdv_combo_b_1(self):
        computer = ChronospatialComputer(10, 2, 2, [6, 5])
        while optcode := computer.next():
            computer.set_operation(optcode[0])
            computer.set_operand(optcode[1])
            computer.apply()

        assert computer.b.get() == 2

    def test_example_custom_bdv_combo_c_99(self):
        computer = ChronospatialComputer(251, 1, 3, [6, 6])
        while optcode := computer.next():
            computer.set_operation(optcode[0])
            computer.set_operand(optcode[1])
            computer.apply()

        assert computer.b.get() == 31

    def test_example_custom_bdv_combo_1(self):
        computer = ChronospatialComputer(250, 2, 50, [6, 1])
        while optcode := computer.next():
            computer.set_operation(optcode[0])
            computer.set_operand(optcode[1])
            computer.apply()

        assert computer.b.get() == 125

    def test_example_custom_cdv_combo_b_1(self):
        computer = ChronospatialComputer(32, 3, 2, [7, 5])
        while optcode := computer.next():
            computer.set_operation(optcode[0])
            computer.set_operand(optcode[1])
            computer.apply()

        assert computer.c.get() == 4

    def test_example_custom_cdv_combo_b_2(self):
        computer = ChronospatialComputer(667, 3, 2, [7, 5])
        while optcode := computer.next():
            computer.set_operation(optcode[0])
            computer.set_operand(optcode[1])
            computer.apply()

        assert computer.c.get() == 83

    def test_example_custom_cdv_combo_c_98(self):
        computer = ChronospatialComputer(1441, 2, 6, [7, 6])
        while optcode := computer.next():
            computer.set_operation(optcode[0])
            computer.set_operand(optcode[1])
            computer.apply()

        assert computer.c.get() == 22

    def test_example_custom_jnz_2(self):
        computer = ChronospatialComputer(0, 999, 888, [3, 2123])
        while optcode := computer.next():
            computer.set_operation(optcode[0])
            computer.set_operand(optcode[1])
            computer.apply()

        assert computer.a.get() == 0
        assert computer.b.get() == 999
        assert computer.c.get() == 888

    def test_example_custom_jnz_3(self):
        computer = ChronospatialComputer(0, 999, 888, [7, 1, 6, 2, 0, 3])
        while optcode := computer.next():
            computer.set_operation(optcode[0])
            computer.set_operand(optcode[1])
            computer.apply()

        assert computer.a.get() == 0
        assert computer.b.get() == 999
        assert computer.c.get() == 888


    def test_example_1(self):
        computer = ChronospatialComputer(1, 1, 1, [2, 6])
        while optcode := computer.next():
            computer.set_operation(optcode[0])
            computer.set_operand(optcode[1])
            computer.apply()

        assert computer.b.get() == 1

    def test_example_2(self):
        actual = []
        computer = ChronospatialComputer(10, 1, 1, [5, 0, 5, 1, 5, 4], actual.append)
        while optcode := computer.next():
            computer.set_operation(optcode[0])
            computer.set_operand(optcode[1])
            computer.apply()

        assert "".join(actual) == '0,1,2,'

    def test_example_3(self):
        actual = []
        computer = ChronospatialComputer(2024, 1, 1, [0, 1, 5, 4, 3, 0], actual.append)
        j = 0
        while optcode := computer.next():
            computer.set_operation(optcode[0])
            computer.set_operand(optcode[1])
            computer.apply()
            j += 1

        assert computer.a.get() == 0
        assert "".join(actual) == '4,2,5,6,7,7,7,7,3,1,0,'


    def test_example_4(self):
        actual = []
        computer = ChronospatialComputer(2024, 29, 1, [1, 7], actual.append)
        while optcode := computer.next():
            computer.set_operation(optcode[0])
            computer.set_operand(optcode[1])
            computer.apply()

        assert computer.b.get() == 26

    def test_example_5(self):
        actual = []
        computer = ChronospatialComputer(2024, 2024, 43690, [4, 0], actual.append)
        while optcode := computer.next():
            computer.set_operation(optcode[0])
            computer.set_operand(optcode[1])
            computer.apply()

        assert computer.b.get() == 44354


    def test_example_1_part_2(self):
        outcome = []
        computer = ChronospatialComputer(117440, 0, 0, [0, 3, 5, 4, 3, 0], outcome.append)
        while optcode := computer.next():
            computer.set_operation(optcode[0])
            computer.set_operand(optcode[1])
            computer.apply()

        actual = ",".join(str(o) for o in outcome)
        assert actual == "0,3,5,4,3,0"


    def test_example_1_part_2_find_self(self):
        outcome = []
        computer = ChronospatialComputer(14, 0, 0, [7, 3, 1, 0], outcome.append)
        while optcode := computer.next():
            computer.set_operation(optcode[0])
            computer.set_operand(optcode[1])
            computer.apply()

        actual = ",".join(str(o) for o in outcome)
        assert actual == "7,3,1,0"
