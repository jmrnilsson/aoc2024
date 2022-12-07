import re
import re
import sys
from enum import Enum
from typing import TypeVar, Generic

from aoc.helpers import locate, build_location, read_lines
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

T = TypeVar('T')


class Node(Generic[T]):
    parent = None
    name: str = None
    size: int = 0
    children: dict[str, Generic[T]] = []
    seen_files: set[tuple[str, int]] = set()

    def __init__(self, name: str, parent: Generic[T]):
        self.parent: Generic[T] = parent
        self.name: str = name
        self.children = {}
        self.files = set()
        self.seen_files: set[str] = set()

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
        if child.name not in self.children.keys():
            self.children[child.name] = child
            return True
        return False

    def add_file(self, file: str, size: int):
        if file not in self.seen_files:
            self.seen_files.add(file)
            self.size += size
            if self.parent is not None:
                self.parent.add_file(file, size)


class Command(Enum):
    LS = r"^\$ ls"
    CD_DOT_DOT = r"^\$ cd .."
    CD = r"^\$ cd (\w+)"
    DIR = r"^dir (\w+)"
    FILE = r"^(\d+) ([\w\.]+)"


def traverse(lines: list[str]):
    root = Node(".", None)
    all_, current_dir, ls_command = {root}, root, False
    for line in lines[1:]:
        if folder := re.findall(Command.CD.value, line):
            current_dir, ls_command = current_dir.children[folder[0]], False
            _ = current_dir.add_child(current_dir)
            all_.add(current_dir)
            continue

        if re.findall(Command.CD_DOT_DOT.value, line):
            current_dir = current_dir.parent
            continue

        if re.findall(Command.LS.value, line):
            ls_command = True
            continue

        if ls_command and (folder := re.findall(Command.DIR.value, line)):
            current_dir.add_child(Node(folder[0], current_dir))
            continue

        if ls_command and (file := re.findall(Command.FILE.value, line)):
            current_dir.add_file(f"{current_dir}/{file[0][1]}", int(file[0][0]))
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
