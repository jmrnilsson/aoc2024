# 🎄 Advent of Code 2022 🎄

- Install: `requirements install -r requirements.txt`
- Run `python3 year_2022/day_07/solve.py`
- Test: Use IDE :)
- Generate README: `python aoc/template.py`

*Note:* For browser automation: https://github.com/jmrnilsson/aoc-watcher


## year_2022/day_08/solve.py

```py
import operator
import sys
from collections import defaultdict
from functools import reduce
from itertools import takewhile, product
from aoc.helpers import locate, build_location, read_lines
from aoc.poll_printer import PollPrinter
from aoc.tools import transpose


def solve_1(input_=None):
    """
    test=58
    expect=1598415
    """
    with open(locate(input_), "r") as fp:
        matrix = [[int(d) for d in list(line)] for line in read_lines(fp)]

    y_range = [("y", n, axis) for n, axis in enumerate(matrix)]
    x_range = [("x", n, axis) for n, axis in enumerate(transpose(matrix))]
    visible_trees = set()
    for axis, _k, _k_range in y_range + x_range:
        for reverse in (False, True):
            ceiling = -1
            k_range = enumerate(_k_range) if not reverse else reversed(list(enumerate(_k_range)))
            for n, tree in k_range:
                coords = (_k, n) if axis == "y" else (n, _k)
                if tree > ceiling:
                    visible_trees.add(coords)
                ceiling = max(tree, ceiling)

    return sum(1 for _ in visible_trees)


def count_while_less_than_add_one(k_range, value):
    total = sum(1 for _ in takewhile(lambda v: v < value, k_range))
    total += 1 if len(k_range) > total else 0
    return total


def splice_at(k_range, n):
    return k_range[slice(n + 1, len(k_range))], list(reversed(k_range[0: n]))


def solve_2(input_=None):
    """
    test=58
    expect=1598415
    """
    scenic = defaultdict(list)

    with open(locate(input_), "r") as fp:
        matrix = [[int(d) for d in list(line)] for line in read_lines(fp)]

    transposed = transpose(matrix)
    x_len, y_len = len(matrix[0]), len(matrix)
    for y, x in product(range(0, y_len), range(0, x_len)):
        x_range, y_range, value = matrix[y],  transposed[x], matrix[y][x]
        down, up = splice_at(y_range, y)
        right, left = splice_at(x_range, x)
        view = [
            count_while_less_than_add_one(right, value),
            count_while_less_than_add_one(left, value),
            count_while_less_than_add_one(down, value),
            count_while_less_than_add_one(up, value)
        ]
        scenic[(y, x)] = view

    scenic_view = {k: reduce(operator.mul, v,  1) for k, v in scenic.items()}
    return max(scenic_view.values())


```
## year_2022/day_07/solve.py

```py
import re
import sys
from enum import Enum
from typing import TypeVar, Generic, Set, Tuple, Dict, List
from aoc.helpers import locate, build_location, read_lines
from aoc.poll_printer import PollPrinter


sys.setrecursionlimit(1100)

T = TypeVar('T')


class Node(Generic[T]):
    parent = None
    name: str = None
    size: int = 0
    children = []
    seen_files: Set[Tuple[str, int]] = set()

    def __init__(self, name: str, parent: Generic[T]):
        self.parent: Generic[T] = parent
        self.name: str = name
        self.children: Dict[str, Generic[T]] = {}
        self.files = set()
        self.seen_files: Set[str] = set()

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


def traverse(lines: List[str]):
    root = Node(".", None)
    all_, current_dir, ls_command = {root}, root, False
    for line in lines[1:]:
        if folder := re.findall(Command.CD.value, line):
            current_dir, ls_command = current_dir.children[folder[0]], False
            _ = current_dir.add_child(current_dir)
            all_.add(current_dir)

        elif re.findall(Command.CD_DOT_DOT.value, line):
            current_dir = current_dir.parent

        elif re.findall(Command.LS.value, line):
            ls_command = True

        elif ls_command and (folder := re.findall(Command.DIR.value, line)):
            current_dir.add_child(Node(folder[0], current_dir))

        elif ls_command and (file := re.findall(Command.FILE.value, line)):
            current_dir.add_file(f"{current_dir}/{file[0][1]}", int(file[0][0]))

        else:
            raise ValueError("Command not parsed")

    return root, all_


def solve_1(input_=None):
    """
    test=95437
    expect=1350966
    """
    with open(locate(input_), "r") as fp:
        lines = read_lines(fp)

    _, nodes = traverse(lines)
    return sum(n.size for n in nodes if n.size <= 100000)


def solve_2(input_=None):
    """
    test=58
    expect=1598415
    """
    with open(locate(input_), "r") as fp:
        lines = read_lines(fp)

    root, all_ = traverse(lines)
    return min(m for m in [n.size for n in all_] if m >= root.size - 70000000 + 30000000)


```
## year_2022/day_06/solve.py

```py
import re
import sys
from more_itertools import sliding_window as win
from aoc.helpers import locate, build_location, read_lines
from aoc.poll_printer import PollPrinter


def solve_1(input_=None):
    with open(locate(input_), "r") as fp:
        lines = read_lines(fp)

    n = 4
    return next(i for i, c in enumerate(win(list(lines[0]), n)) if len(set(c)) == n) + n


def solve_2(input_=None):
    with open(locate(input_), "r") as fp:
        lines = read_lines(fp)

    n = 14
    return next(i for i, c in enumerate(win(list(lines[0]), n)) if len(set(c)) == n) + n


```
## year_2022/day_05/solve.py

```py
import re
import re
import sys
from aoc import tools
from aoc.helpers import locate, build_location, read_lines
from aoc.poll_printer import PollPrinter


def stack_inverted(input_, start):
    head = []
    with open(locate(input_), "r") as fp_0:
        head_lines = list(read_lines(fp_0))[:start]
        for line in head_lines:
            if not line.strip():
                continue

            plain_line = line.replace("[", " ").replace("]", " ").replace("\n", "")
            head.append(list(plain_line))

    matrix_padded = tools.matrix_pad(head, " ")
    transpose_head = tools.transpose(matrix_padded)
    reverse_matrix_head = tools.reverse_matrix_x(transpose_head)
    stacks = {int(line[0]): [li for li in line[1:] if li.strip()] for line in reverse_matrix_head if "".join(line).strip()}
    return stacks


def solve_1(input_=None):
    start = 4 if "test" in input_ else 9
    crates = stack_inverted(input_, start)

    step = 0
    with open(locate(input_), "r") as fp:
        lines = [l.strip() for l in read_lines(fp) if l.strip()][start:]
        for line in lines:
            n, from_, to = [int(d) for d in re.findall(r'\d+', line)]
            while n > 0:
                crates[to].append(crates[from_].pop())
                n -= 1
            step += 1

    return str.join("", [li[-1] for li in crates.values()])


def solve_2(input_=None):
    start = 4 if "test" in input_ else 9
    crates = stack_inverted(input_, start)

    step = 0
    with open(locate(input_), "r") as fp:
        lines = [l.strip() for l in read_lines(fp) if l.strip()][start:]
        for line in lines:
            n, from_, to = [int(d) for d in re.findall(r'\d+', line)]
            swap = []
            while 1:
                if n < 1:
                    break

                swap.insert(0, crates[from_].pop())
                n -= 1

            for s in swap:
                crates[to].append(s)

            step += 1

    return str.join("", [li[-1] for li in crates.values()])


```
## year_2022/day_04/solve.py

```py
import re
import sys
from aoc.helpers import locate, build_location, read_lines
from aoc.poll_printer import PollPrinter


def solve_1(input_=None):
    """
    test=2
    expect=518
    """
    total = 0

    with open(locate(input_), "r") as fp:
        for line in read_lines(fp):
            current, other = [[int(value) for value in i.split("-")] for i in re.findall(r"\d+-\d+", line)]
            if current[0] >= other[0] and current[1] <= other[1] or other[0] >= current[0] and other[1] <= current[1]:
                total += 1

    return total


def solve_2(input_=None):
    """
    test=4
    expect=909
    """
    total = 0

    with open(locate(input_), "r") as fp:
        for line in read_lines(fp):
            both = [(int(i.split("-")[0]), int(i.split("-")[1])) for i in re.findall(r"\d+-\d+", line)]
            current, other = both
            current_min, current_max, other_min, other_max = current[0], current[1], other[0], other[1]

            if set(range(current_min, current_max + 1)) & set(range(other_min, other_max + 1)):
                if current_min <= other_min <= current_max or current_min <= other_max <= current_max:
                    total += 1
                elif other_min <= current_min <= other_max or other_min <= current_max <= other_max:
                    total += 1

    return total


```
## year_2022/day_03/solve.py

```py
import re
import re
import sys
from more_itertools import chunked
from aoc.helpers import locate, build_location
from aoc.poll_printer import PollPrinter


def solve_1(input_=None):
    """
    test=157
    expect=8105
    """
    seed, value = [], 0
    with open(locate(input_), "r") as fp:
        for line in (line for line in fp.readlines() if line != "\n"):
            seed.append([i for i in re.findall(r"\w+", line)][0])

    for rucksacks, length in ((r, len(r) // 2) for r in seed):
        char, = set(rucksacks[:length]) & set(rucksacks[length:])
        value += ord(char) - 96 if char == char.lower() else ord(char) - 38

    return value


def solve_2(input_=None):
    """
    test=70
    expect=2363
    """
    seed, value = [], 0,
    with open(locate(input_), "r") as fp:
        for chunk in chunked((line for line in fp.readlines() if line != "\n"), 3):
            seed.append([i for i in re.findall(r"\w+", line)][0] for line in chunk)

    for rucksacks in seed:
        char, = set.intersection(*[set(r) for r in rucksacks])
        value += ord(char) - 96 if char == char.lower() else ord(char) - 38

    return value


```
## year_2022/day_03/solve.js

```javascript
const { timeEnd } = require('node:console');
const {readFile} = require('node:fs/promises');

/*
# ICE
_default_puzzle_input = "year_2022/day_01/puzzle.txt"
_default_test_input = "year_2022/day_01/test.txt"

puzzle_input = build_location(__file__, "puzzle.txt")
test_input = build_location(__file__, "test.txt")
test_input_2 = build_location(__file__, "test_2.txt")
*/

const test_input = "aoc\\year_2022\\day_03_extras_javascript\\test.txt";
const puzzle_input = "aoc\\year_2022\\day_03_extras_javascript\\puzzle.txt";

const challenge_solve_1 = 157
const challenge_solve_2 = 70


function intersect(left, right){
    return new Set([...left].filter(x => right.has(x)));
}

function* chunked(arr, n) {
    for (let i = 0; i < arr.length; i += n) {
        yield arr.slice(i, i + n);
    }
}

/*
test=157
expect=8105
*/ 
async function solve1(input){
    let [seed, value] = [[], 0]

    for (const line of (await readFile(input, { encoding: 'utf8' })).split('\n')){
        if (!line) continue;
        seed.push([line, line.length / 2])
    }

    for (const [rs, len] of seed){
        let [char] = intersect(new Set(rs.split('').slice(0, len)), new Set(rs.split('').slice(len, len*2)));
        value += char == char.toLowerCase() ? char.charCodeAt() - 96 : char.charCodeAt() - 38;
    }

    return value;
}

/*
test=70
expect=2363
*/ 
async function solve2(input){
    let [seed, value] = [[], 0]

    const lines = (await readFile(input, { encoding: 'utf8' })).split('\n');
    for (const chunk of chunked(lines.slice(0, lines.length -2), 3)){
        seed.push([chunk])
    }

    for (const [rucksack] of seed){
        var [tail, head] = [rucksack.slice(1, rucksack.length), new Set(rucksack[0])];
        let [char] = tail.reduce((a, v) => intersect(a, new Set(v)), head);
        value += char == char.toLowerCase() ? char.charCodeAt() - 96 : char.charCodeAt() - 38;
    }

    return value;
}

async function timed(func_name, ...args){
    const func = eval(func_name);
    ts = performance.now();
    const answer = await func(...args)
    ms = (performance.now() - ts).toFixed(4);
    const name = args[0].match("(test|puzzle)", 'i')[1];
    return `${func_name} - ${name.padEnd(6," ")} (${ms}ms):`.padEnd(33," ") + answer;
}


async function start(argv) {
    console.log(await timed("solve1", test_input));
    console.log(await timed("solve1", puzzle_input));
    console.log(await timed("solve2", test_input));
    console.log(await timed("solve2", puzzle_input));
    console.log("\nDone!");
    process.exit();
}

start(process.argv);
```
## year_2022/day_02/solve.py

```py
import re
import sys
from aoc.helpers import locate, build_location, read_lines
from aoc.poll_printer import PollPrinter


def solve_1(input_=None):
    feed = []

    lookup = {
        "A": "ROCK",
        "X": "ROCK",
        "B": "PAPER",
        "Y": "PAPER",
        "Z": "SCISSORS",
        "C": "SCISSORS",
    }

    with open(locate(input_), "r") as fp:
        for line in read_lines(fp):
            other, me = line.split(" ")
            me_ = me.replace("\n", "")
            feed.append((lookup[other], lookup[me_]))

    point = 0
    for other, me in feed:
        if other == "ROCK" and me == "PAPER":
            point += 6
        if other == "PAPER" and me == "SCISSORS":
            point += 6
        if other == "SCISSORS" and me == "ROCK":
            point += 6

        if other == me:
            point += 3

        if me == "ROCK":
            point += 1
        if me == "PAPER":
            point += 2
        if me == "SCISSORS":
            point += 3

    return point


def solve_2(input_=None):
    feed = []
    lookup = {
        "A": "ROCK",
        "X": "ROCK",
        "B": "PAPER",
        "Y": "PAPER",
        "Z": "SCISSORS",
        "C": "SCISSORS",
    }

    with open(locate(input_), "r") as fp:
        for line in read_lines(fp):
            other, me = line.split(" ")
            me_ = me.replace("\n", "")
            me__ = None
            other_ = lookup[other]
            if me_ == "X":
                if other_ == "ROCK":
                    me__ = "SCISSORS"
                if other_ == "SCISSORS":
                    me__ = "PAPER"
                if other_ == "PAPER":
                    me__ = "ROCK"
            elif me_ == "Y":
                me__ = other_
            else:
                if other_ == "ROCK":
                    me__ = "PAPER"
                if other_ == "SCISSORS":
                    me__ = "ROCK"
                if other_ == "PAPER":
                    me__ = "SCISSORS"
            feed.append((other_, me__))

    point = 0
    for other, me in feed:
        if other == "ROCK" and me == "PAPER":
            point += 6
        if other == "PAPER" and me == "SCISSORS":
            point += 6
        if other == "SCISSORS" and me == "ROCK":
            point += 6

        if other == me:
            point += 3

        if me == "ROCK":
            point += 1
        if me == "PAPER":
            point += 2
        if me == "SCISSORS":
            point += 3

    return point


```
## year_2022/day_01/solve.py

```py
import re
import sys
from aoc.helpers import locate, build_location
from aoc.poll_printer import PollPrinter


def solve_1(input_=None):
    feed = [0]

    with open(locate(input_), "r") as fp:
        for line in fp.readlines():
            if line == "\n":
                feed.append(0)
                continue
            feed[-1] += int(line)

    return max(feed)


def solve_2(input_=None):
    feed = [0]

    with open(locate(input_), "r") as fp:
        for line in fp.readlines():
            if line == "\n":
                feed.append(0)
                continue
            feed[-1] += int(line)

    feed.sort(key=lambda c: c)

    return sum(feed[-3:])


```

