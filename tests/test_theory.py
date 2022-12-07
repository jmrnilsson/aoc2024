import itertools
import unittest

from aoc import theory_set
from aoc.helpers import build_location


class TestLocation(unittest.TestCase):

    def test_theory_intersection_any(self):
        seen = set()
        expected = set()
        actual = set()
        a = range(0, 25)
        for a, b, c, d in itertools.product(a, repeat=4):
            if a > b or c > d:
                continue
            left, right = (a, b), (c, d)
            if (left, right) in seen:
                continue

            intersects = any(set(range(*left)).intersection(set(range(*right))))
            intersection_any_theory = theory_set.intersection_any(left, right)
            if intersection_any_theory:
                actual.add((left, right))
            if intersects:
                expected.add((left, right))

            seen.add((left, right))

        self.assertEqual(expected, actual)

    def test_theory_count_intersection_count(self):
        pass
