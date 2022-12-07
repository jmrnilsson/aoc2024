import unittest

from aoc.helpers import build_location


class TestLocation(unittest.TestCase):

    def test_build_location(self):
        actual = build_location(__file__, "random.txt")
        self.assertIn("aoc\\tests\\random.txt", actual)
