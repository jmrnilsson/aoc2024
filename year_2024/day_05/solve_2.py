import sys
from dataclasses import dataclass
from typing import List, Tuple, Set

from numpy.ma.extras import median

from aoc.helpers import locate, build_location, read_lines
from aoc.printer import print_, get_meta_from_fn


sys.setrecursionlimit(3000)

_default_puzzle_input = "year_2024/day_01/puzzle.txt"
_default_test_input = "year_2024/day_01/test.txt"

puzzle_input = build_location(__file__, "puzzle.txt")
test_input = build_location(__file__, "test.txt")
test_input_2 = build_location(__file__, "test_2.txt")


@dataclass
class Rule:
    left: int
    right: int
    __hash: int

    def __init__(self, left: int, right: int):
        self.left = left
        self.right = right
        self.__hash = hash(tuple([self.left, self.right]))

    def __hash__(self):
        return self.__hash

class SortAutomaton:
    seen: Set[Tuple[int, ...]]
    sequence: List[int]
    unique = Set[int]

    def __init__(self, starting_sequence: Tuple[int, ...]):
        self.seen = set()
        self.seen.add(starting_sequence)
        self.sequence = list(starting_sequence)
        self.unique = set(starting_sequence)

    def process(self, rule: Rule):
        if rule.left in self.unique and rule.right in self.unique:
            left_index = self.sequence.index(rule.left)
            right_index = self.sequence.index(rule.right)
            if left_index > right_index:  # swap
                # print(f"sequence: {self.sequence}, swap: {rule.left} <-> {rule.right} for index {self.index}")
                self.sequence[right_index] = rule.left
                self.sequence[left_index] = rule.right
                # print(f"sequence: {self.sequence} after.")

    def get_middle_number_or_zero_if_unchanged(self):
        if len(self.seen) < 2:
            return 0

        half = len(self.sequence) // 2
        middle_number = self.sequence[half]
        return middle_number

    def is_accepting(self):
        checked: Tuple[int, ...] = tuple(self.sequence)
        accept = checked in self.seen
        self.seen.add(checked)
        return accept

def solve(__input=None):
    """
    :challenge: 123
    :expect: 6897
    """
    rules: List[Rule] = []
    updates: List[Tuple[int, ...]] = []
    with open(locate(__input), "r") as fp:
        for ind, line in enumerate(read_lines(fp)):
            if "|" in line:
                left, right = [int(l) for l in line.split("|")]
                rule = Rule(left, right)
                rules.append(rule)
            elif "," in line:
                update = tuple([int(l) for l in line.split(",")])
                updates.append(update)

    i = 0
    total = 0
    for update in updates:
        automaton = SortAutomaton(update)
        i += 1

        while 1:
            for rule in rules:
                automaton.process(rule)

            if automaton.is_accepting():
                break

        total += automaton.get_middle_number_or_zero_if_unchanged()

    return total


if __name__ == "__main__":
    challenge = get_meta_from_fn(solve, "challenge")
    expect = get_meta_from_fn(solve, "expect")
    print_(solve, test_input, puzzle_input, challenge, expect)
