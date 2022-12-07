import heapq
import json
import operator
import re
import sys
from collections import Counter, OrderedDict, defaultdict

import numpy as np

from aoc import tools
from aoc.helpers import timing, locate, Printer, build_location, test_nocolor, puzzle_nocolor, read_lines
from aoc.poll_printer import PollPrinter

# ICE
_default_puzzle_input = "year_2022/day_01/puzzle.txt"
_default_test_input = "year_2022/day_01/test.txt"

puzzle_input = build_location(__file__, "puzzle.txt")
test_input = build_location(__file__, "test.txt")
test_input_2 = build_location(__file__, "test_2.txt")

challenge_solve_1 = 95437
challenge_solve_2 = "exception"

sys.setrecursionlimit(1100)


class Node(object):
    parent = None
    name = None
    size = 0
    children = []
    seen_files = set()

    def __init__(self, name: str, parent: object):
        self.parent = parent
        self.name: str = name
        self.children = []
        self.files = set()
        self.seen_files = set()

    def __hash__(self):
        return hash(self.get_path())

    def __eq__(self, other):
        return self.get_path() == other.get_path()

    def __repr__(self):
        return f"{self.get_path()}: {self.size}"

    def get_path(self):
        path = []
        node = self
        while node is not None:
            path.append(node.name)
            node = node.parent
        return "/".join(reversed(path))

    def add_child(self, child):
        if child.name not in set(c.name for c in self.children):
            self.children.append(child)
            return True
        return False

    def add_file(self, file: str, size: int):
        if file not in self.seen_files:
            self.seen_files.add(file)
            self.size += size
            if self.parent is not None:
                self.parent.add_file(file, size)


def traverse(lines: list[str]):
    count = 1
    root = Node(".", None)
    all_ = set()
    all_.add(root)
    seen_ls = set()
    current_dir = root
    ls_operation = False
    n = 0
    for line in lines[1:]:
        n += 1
        dir_ = re.findall(r"^\$ cd (\w+)", line)
        if dir_:  # cd
            folder_name = dir_[0]
            swap, = [f for f in current_dir.children if f.name == folder_name]
            added = current_dir.add_child(swap)
            if not added:
                all_.add(swap)
            current_dir = swap
            ls_operation = False
            count += 1
            continue

        cd_dot_dot = re.findall(r"^\$ cd ..", line)
        if cd_dot_dot:  # cd..
            swap = current_dir.parent
            if swap:
                current_dir = swap
            continue

        ls = re.findall(r"^\$ ls", line)
        if ls:  # ls
            ls_operation = True if current_dir.get_path() not in seen_ls else False
            seen_ls.add(current_dir.name)
            continue

        if ls_operation:
            folder = re.findall(r"^dir (\w+)", line)
            if folder:
                fold = Node(folder[0], current_dir)
                if fold not in current_dir.children:
                    current_dir.add_child(fold)
                continue

            file = re.findall(r"^(\d+) ([\w\.]+)", line)
            if file:
                size, name = file[0]
                current_dir.add_file(f"{current_dir}/{name}", int(size))
                continue
    return root, all_


def solve_1(input_=None):
    """
    test=95437
    expect=1350966
    """
    with open(locate(input_), "r") as fp:
        lines = read_lines(fp)

    _, all_ = traverse(lines)

    return sum(n.size for n in all_ if n.size <= 100000)


def solve_2(input_=None):
    """
    test=58
    expect=1598415
    """
    is_test = 1 if "test" in input_ else 0

    with open(locate(input_), "r") as fp:
        lines = read_lines(fp)

    root, all_ = traverse(lines)

    root_sum = root.size
    all_sums = {n.get_path(): n.size for n in all_}
    if is_test:
        assert all_sums == {".": 48381165, "./d": 24933642, "./a/e": 584, "./a": 94853}
    required = root_sum - 70000000 + 30000000
    allowed = [m for m in all_sums.values() if m >= required]
    return min(allowed)


if __name__ == "__main__":
    """
    PWSH
    while ($true) {python -m aoc.year_2022.day_03.solve -poll; start-sleep -seconds 1} 

    BASH
    watch -n 3 python3 -m aoc.year_2021.day_03.day -p
    """
    poll_printer = PollPrinter(solve_1, solve_2, challenge_solve_1, challenge_solve_2, test_input, puzzle_input)
    args = sys.argv[1:]
    if not args:
        poll_printer.print_timed()
    elif re.match("^-poll$|^-p$", args[0]):
        poll_printer.poll_print()
    elif re.match("^-json1$|^-j1$", args[0]):
        poll_printer.poll_json_1()
    elif re.match("^-json2$|^-j2$", args[0]):
        poll_printer.poll_json_2()
