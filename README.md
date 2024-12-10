# 🎄 Advent of Code 2024 😅

## Install
```bash
python.exe -m venv .venv
python.exe -m pip install --upgrade pip
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Usage
- Run `python3 year_2023/day_07/solve.py`
- Generate README: `python aoc/template.py`


## year_2024\day_10\solve_2.py

```py
import sys
import numpy as np
from aoc.helpers import locate, build_location, read_lines
from aoc.printer import get_meta_from_fn, print_
from year_2024.day_10.solve_1 import HikingAutomaton


sys.setrecursionlimit(30_000)

def solve(__input=None):
    """
    :challenge: 81
    :expect: 1034
    """
    lines = []
    with open(locate(__input), "r") as fp:
        for line in read_lines(fp):
            a = list(map(int, list(line)))
            lines.append(a)

    grid = np.matrix(lines)
    starting_positions = [[(int(y), int(x))] for y, x in np.argwhere(grid == 0)]

    hiker = HikingAutomaton(starting_positions, grid)
    while not hiker.is_accepting():
        hiker.walk()

    return hiker.rat()

```
## year_2024\day_10\solve_1.py

```py
import sys
from copy import deepcopy
from typing import List, Tuple, Generator
import numpy as np
from aoc.helpers import locate, build_location, read_lines
from aoc.printer import get_meta_from_fn, print_


sys.setrecursionlimit(30_000)

class HikingAutomaton:
    trails: List[List[Tuple[int, int]]]
    done: List[List[Tuple[int, int]]]

    def __init__(self, starting_pos: List[List[Tuple[int, int]]], grid):
        self.trails = list(starting_pos)
        self.done = []
        self.grid = grid

    def in_bound(self, y: int, x: int) -> bool:
        return -1 < y < self.grid.shape[0] and -1 < x < self.grid.shape[1]

    def _step(self, y, x) -> Generator[Tuple[int, int, int], None, None]:
        steps = (
            (y - 1, x + 0),
            (y + 0, x + 1),
            (y + 1, x + 0),
            (y + 0, x + -1)
        )
        for step in steps:
            if self.in_bound(*step) and (value := self.grid[step]) == self.grid[y, x] + 1:
                yield *step, value

    def walk(self):
        trails = list(self.trails)
        self.trails.clear()
        for n, trail_ in enumerate(trails):
            last_step = trail_[-1]
            for y, x, value in self._step(*last_step):
                trail = deepcopy(trail_)
                trail.append((y, x))
                if value == 9:
                    self.done.append(trail)
                else:
                    self.trails.append(trail)

    def score(self):
        head_and_tails = {
            (trail[0], trail[-1])
            for trail in self.done
        }
        return len(head_and_tails)

    def rat(self):
        return len(self.done)

    def is_accepting(self):
        return len(self.trails) == 0

def solve(__input=None):
    """
    :challenge: 36
    :expect: 459
    """
    lines = []
    with open(locate(__input), "r") as fp:
        for line in read_lines(fp):
            a = list(map(int, list(line)))
            lines.append(a)

    grid = np.matrix(lines)
    starting_positions = [[(int(y), int(x))] for y, x in np.argwhere(grid == 0)]

    hiker = HikingAutomaton(starting_positions, grid)
    while not hiker.is_accepting():
        hiker.walk()

    return hiker.score()

```
## year_2024\day_09\solve_2.py

```py
import sys
from typing import List
from more_itertools import chunked
from aoc.helpers import locate, build_location, read_lines
from aoc.printer import get_meta_from_fn, print_


sys.setrecursionlimit(30_000)

class Block:
    """
    Block of disk marshalled by reference.
    """

    def __init__(self, number: int, size: int, free: int = 0):
        self.number = number
        self.size = size
        self.free = free

    def __str__(self):
        return f"Block({self.number}, {self.size}, {self.free})"

    def __repr__(self):
        return str(self)

    def __hash__(self):
        return self.number

def reshape(_blocks: List[Block]): # Generator[Tuple[int, int, int], None, None]
    blocks = list(_blocks)
    i = len(_blocks) - 1
    while i > 1:
        candidate = _blocks[i]

        if not candidate.size:
            continue

        index_of_candidate = blocks.index(candidate)
        for j in range(0, index_of_candidate, 1):
            block = blocks[j]

            if block.free >= candidate.size:
                blocks.insert(j + 1, Block(candidate.number, candidate.size, block.free - candidate.size))
                candidate.free = candidate.size + candidate.free
                candidate.size = 0
                block.free = 0
                break

        i -= 1

    return blocks

def solve(__input=None):
    """
    :challenge: 2858
    :expect: 6307653242596
    """
    with open(locate(__input), "r") as fp:
        lines = [int(i) for i in list(read_lines(fp)[0])]

    blocks: List[Block] = [
        Block(number, *chunk) for number, chunk in enumerate(chunked(lines, 2))
    ]

    index = 0
    total = 0
    for v in reshape(blocks):
        total += sum(i * v.number for i in range(index, index + v.size))
        index += v.size + v.free

    return total

```
## year_2024\day_09\solve_1.py

```py
import sys
from typing import List, Tuple, Generator
from more_itertools import chunked
from aoc.helpers import locate, build_location, read_lines
from aoc.printer import get_meta_from_fn, print_


sys.setrecursionlimit(30_000)

def reshape(blocks: List[Tuple[int, int, int]]) -> Generator[Tuple[int, int], None, None]:
    if len(blocks) < 1:
        return

    number_head, size_head, free_head = blocks[0]
    if size_head:
        yield number_head, size_head

    if free_head < 1 < len(blocks):
        for res in reshape(blocks[1:]):
            yield res

    elif free_head and len(blocks) > 1:
        number_tail, size_tail, free_tail = blocks[-1]

        if size_tail == free_head:
            yield number_tail, size_tail
            for res in reshape(blocks[1:-1]):
                yield res

        if free_head > size_tail:
            yield number_tail, size_tail
            new_blocks = [(0, 0, free_head - size_tail)] + blocks[1:-1]
            for res in reshape(new_blocks):
                yield res

        if size_tail > free_head:
            yield number_tail, free_head
            new_blocks = blocks[1:-1] + [(number_tail, size_tail - free_head, free_tail)]

            for res in reshape(new_blocks):
                yield res

def solve(__input=None):
    """
    :challenge: 1928
    :expect: 6283170117911
    """
    with open(locate(__input), "r") as fp:
        lines = [int(i) for i in list(read_lines(fp)[0])]

    to_block = lambda chunk_: (chunk_[0], chunk_[1] if len(chunk_) > 1 else 0)
    blocks = [(number, *to_block(chunk)) for number, chunk in enumerate(chunked(lines, 2))]
    defragmented = reshape(blocks)

    j = 0
    total = 0
    for number, size in defragmented:
        total += sum(i * number for i in range(j, j + size))
        j += size

    return total

```
## year_2024\day_08\solve_2.py

```py
import itertools
import operator
import sys
from collections import defaultdict
from typing import List, Tuple, Set, DefaultDict, Callable, Generator
from aoc.helpers import locate, build_location, read_lines
from aoc.printer import get_meta_from_fn, print_


sys.setrecursionlimit(3000)

def find_antinode(shape: Tuple[int, int], edge, y_real: int, x_real, op: Callable) -> Generator[Tuple[int, int], None, None]:
    width, height = shape
    for k in range(0, max(*shape)):
        antinode = op(edge[0], y_real * k), op(edge[1], x_real * k)
        if not -1 < antinode[0] < height:
            break
        if not -1 < antinode[1] < width:
            break
        yield antinode
        # antinodes[antenna_name].add(an)

def solve(__input=None):
    """
    :challenge: 34
    :expect: 951
    """
    antennas: DefaultDict[Set[Tuple[int, int]]] = defaultdict(set)
    seed: List[List[str]] = []
    with open(locate(__input), "r") as fp:
        for ind, line in enumerate(read_lines(fp)):
            seed.append(list(line))

    shape = len(seed[0]), len(seed)

    for y, row in enumerate(seed):
        for x, cell in enumerate(row):
            match cell:
                case ".": continue
                case "#": continue
                case _: antennas[cell].add((y, x))

    antinodes: DefaultDict[Set[Tuple[int, int]]] = defaultdict(set)
    for antenna, coords in antennas.items():
        for a, b in itertools.combinations(coords, 2):
            y_displacement, x_displacement = a[0] - b[0], a[1] - b[1]
            left_most, right_most = sorted([a, b], key=lambda entry: entry[1])
            if x_displacement < 0:
                for antinode in find_antinode(shape, left_most, y_displacement, x_displacement, operator.add):
                    antinodes[antenna].add(antinode)

                for antinode in find_antinode(shape, right_most, y_displacement, x_displacement, operator.sub):
                    antinodes[antenna].add(antinode)
            else:
                for antinode in find_antinode(shape, right_most, y_displacement, x_displacement, operator.add):
                    antinodes[antenna].add(antinode)

                for antinode in find_antinode(shape, left_most, y_displacement, x_displacement, operator.sub):
                    antinodes[antenna].add(antinode)

    width, height = shape
    unique_antinodes = {
        (y, x)
        for antenna, coords in antinodes.items()
        for y, x in coords
        if -1 < y < height and -1 < x < width
    }

    return len(unique_antinodes)

```
## year_2024\day_08\solve_1.py

```py
import itertools
import sys
from collections import defaultdict
from typing import List, Tuple, Set, DefaultDict
from aoc.helpers import locate, build_location, read_lines
from aoc.printer import get_meta_from_fn, print_


sys.setrecursionlimit(3000)

def solve(__input=None):
    """
    :challenge: 14
    :expect: 254
    """
    antennas: DefaultDict[Set[Tuple[int, int]]] = defaultdict(set)
    seed: List[Tuple[str]] = []
    with open(locate(__input), "r") as fp:
        for ind, line in enumerate(read_lines(fp)):
            seed.append(list(line))

    width, height = len(seed[0]), len(seed)

    for y, row in enumerate(seed):
        for x, cell in enumerate(row):
            match cell:
                case ".": continue
                case "#": continue
                case _: antennas[cell].add((y, x))

    antinodes: DefaultDict[Set[Tuple[int, int]]] = defaultdict(set)
    for antenna, coords in antennas.items():
        for a, b in itertools.combinations(coords, 2):
            y_displacement, x_displacement = a[0] - b[0], a[1] - b[1]
            left_most, right_most = sorted([a, b], key=lambda entry: entry[1])

            if x_displacement < 0:
                antinode_0 = left_most[0] + y_displacement, left_most[1] + x_displacement
                antinodes[antenna].add(antinode_0)
                antinode_1 = right_most[0] - y_displacement, right_most[1] - x_displacement
                antinodes[antenna].add(antinode_1)
            else:
                antinode_2 = right_most[0] + y_displacement, right_most[1] + x_displacement
                antinodes[antenna].add(antinode_2)
                antinode_3 = left_most[0] - y_displacement, left_most[1] - x_displacement
                antinodes[antenna].add(antinode_3)

    unique_antinodes = {
        (y, x)
        for antenna, coords in antinodes.items()
        for y, x in coords
        if -1 < y < height and -1 < x < width
    }

    return len(unique_antinodes)

```
## year_2024\day_07\solve_2.py

```py
import operator
import re
import sys
from functools import reduce
from typing import List, Callable, Tuple
from aoc.helpers import locate, build_location, read_lines
from aoc.printer import print_, get_meta_from_fn


sys.setrecursionlimit(3000)

def concat(left: int, right: int):
    quotient = right
    while quotient := quotient // 10:
        left *= 10

    return left * 10 + right

def computable(acc: Tuple[int, List[int]], value: int):
    answer, seed = acc
    return answer, [v
        for s in seed
        for v in (
            operator.mul(s, value),
            operator.add(s, value),
            concat(s, value)
        )
        if answer >= v
     ]

def calibration_possible(row: List[int]):
    answer, first, *rest = row
    while 1:
        _, remainder = reduce(computable, rest, (answer, [first]))
        if answer in remainder:
            return answer
        elif not remainder:
            return 0
        rest = remainder

def solve(__input=None, prefer_fn: Callable | None  = None):
    """
    :challenge: 11387
    :expect: 169122112716571
    """
    width = 0
    rows: List[List[int]] = []
    with open(locate(__input), "r") as fp:
        for _, line in enumerate(read_lines(fp)):
            items = list(map(int, re.findall(r"\d+", line)))
            rows.append(items)
            width = max(width, len(items))

    return sum(calibration_possible(r) for r in rows)

```
## year_2024\day_07\solve_1.py

```py
import operator
import re
import sys
from typing import List
from aoc.helpers import locate, build_location, read_lines
from aoc.printer import print_, get_meta_from_fn


sys.setrecursionlimit(3000)

class WalkAutomaton:
    expect: int
    state: List[int]

    def __init__(self, expect_: int, start: int):
        self.expect = expect_
        self.state = [start]

    def eval(self, value: int):
        ops = [(operator.mul, "*"), (operator.add, "+")]
        previous_values = list(self.state)
        values = []
        for previous_value in previous_values:
            for op, _ in ops:
                values.append(op(previous_value, value))
        self.state = values

    def is_accepting(self):
        for v in self.state:
            if v == self.expect:
                return True

        return False

def solve(__input=None):
    """
    :challenge: 3749
    :expect: 1545311493300
    """
    lines: List[List[int]] = []
    with open(locate(__input), "r") as fp:
        for _, line in enumerate(read_lines(fp)):
            items = list(map(int, re.findall(r"\d+", line)))
            lines.append(items)

    total = 0
    for row in lines:
        a = WalkAutomaton(row[0], row[1])
        for value in row[2:]:
            a.eval(value)

        if a.is_accepting():
            total += a.expect

    return total

```
## year_2024\day_06\solve_2.py

```py
import sys
from enum import Enum
from typing import List, Tuple, Set
import numpy as np
from aoc.helpers import locate, build_location, read_lines
from aoc.printer import print_, get_meta_from_fn


sys.setrecursionlimit(3000)

class Accept(Enum):
    Pending = 0
    Stuck = 1
    Outside = 2

class WalkAutomaton:
    pos: Tuple[int, int]
    dir: int
    shape_set = Set[int]
    seen = Set[Tuple[int, int, int]] # grid y, x, dir
    steps = int
    outside: bool
    obstacle: Tuple[int, int]

    def __init__(self, starting_position: Tuple[int, int], starting_direction: int, grid, obstacle: Tuple[int, int]):
        self.pos = starting_position
        self.dir = starting_direction
        self.grid = grid
        self.shape_set = set(grid.shape)
        self.seen = set()
        self.steps = 0
        self.outside = False
        self.obstacle = obstacle

    def _step(self) -> Tuple[int, int]:
        y, x = self.pos
        match self.dir:
            case 0: return y - 1, x + 0
            case 90: return y + 0, x + 1
            case 180: return y + 1, x + 0
            case 270: return y + 0, x + -1

        raise TypeError("What's going on here!")

    def _turn(self):
        new_dir = (90 + self.dir) % 360
        self.dir = new_dir

    def walk(self):
        self.steps += 1
        new_pos = self._step()
        if outside := -1 in new_pos or self.shape_set.intersection(set(new_pos)):
            self.outside = outside
            return

        if self.grid[new_pos] == 1 or new_pos == self.obstacle:  # obstructed
            self._turn()
        else:
            self.pos = self._step()
            self.grid[self.pos] = 3

    def is_accepting(self) -> Accept:
        if self.outside:
            return Accept.Outside

        hashable = tuple([*self.pos, self.dir])
        if hashable in self.seen:
            return Accept.Stuck

        self.seen.add(hashable)
        return Accept.Pending

def to_int(text: str):
    match text:
        case "#": return 1
        case "^": return 2
        case _: return 0

def solve(__input=None):
    """
    :challenge: 6
    :expect: 1482
    :notes: A clever solution required or a better brute. Ideas include memoization, warp, traces or GPUs.
    """
    _grid: List[List[int]] = []
    with open(locate(__input), "r") as fp:
        for line in read_lines(fp):
            _grid.append([to_int(character) for character in list(line)])

    grid = np.matrix(_grid, dtype=np.int8)
    starting_position = tuple(np.argwhere(grid == 2)[0])
    candidate_obstruction_positions = [tuple(o) for o in np.argwhere(grid == 0)]
    grid[starting_position] = 3

    total = 0
    for i, obstacle in enumerate(candidate_obstruction_positions):
        if i % 75 == 0:
            print(f"{i} of {len(candidate_obstruction_positions)}")

        walk = WalkAutomaton(starting_position, 0, grid, obstacle)
        while 1:
            walk.walk()
            if accept := walk.is_accepting():
                if accept == Accept.Stuck:
                    total += 1
                    break
                if accept == Accept.Outside:
                    break

    return total

```
## year_2024\day_06\solve_1.py

```py
import sys
from typing import List, Tuple, Set
import numpy as np
from aoc.helpers import locate, build_location, read_lines
from aoc.printer import get_meta_from_fn, print_


sys.setrecursionlimit(3000)

class WalkAutomaton:
    pos: Tuple[int, int]
    dir: int
    shape_set = Set[int]
    seen = Set[Tuple[int, int, int]] # grid y, x, dir
    steps = int
    outside: bool

    def __init__(self, starting_pos: Tuple[int, int], starting_direction: int, grid):
        self.pos = starting_pos
        self.dir = starting_direction
        self.grid = grid
        self.shape_set = set(grid.shape)
        self.seen = set()
        self.steps = 0
        self.outside = False

    def _step(self) -> Tuple[int, int]:
        y, x = self.pos
        match self.dir:
            case 0: return y - 1, x + 0
            case 90: return y + 0, x + 1
            case 180: return y + 1, x + 0
            case 270: return y + 0, x + -1

        raise TypeError("What's going on here!")

    def _turn(self):
        new_dir = (90 + self.dir) % 360
        self.dir = new_dir

    def walk(self):
        self.steps += 1
        new_pos = self._step()
        if outside := -1 in new_pos or self.shape_set.intersection(set(new_pos)):
            self.outside = outside
            return

        if self.grid[new_pos] == 1:  # obstructed
            self._turn()
        else:
            self.pos = self._step()
            self.grid[self.pos] = 3

    def is_accepting(self):
        return self.outside

def to(v: str):
    if v == "#":
        return 1
    elif v == "^":
        return 2
    else:
        return 0

def solve_1(__input=None):
    """
    :challenge: 41
    :expect: 4973
    """
    _grid: List[List[int]] = []
    with open(locate(__input), "r") as fp:
        for ind, line in enumerate(read_lines(fp)):
            l = [to(l) for l in list(line)]
            _grid.append(l)

    grid = np.matrix(_grid, dtype=np.int8)
    starting_position = tuple(np.argwhere(grid == 2)[0])
    grid[starting_position] = 3

    walk = WalkAutomaton(starting_position, 0, grid)
    while 1:
        walk.walk()
        if walk.is_accepting():
            break

    return sum(1 for _ in np.argwhere(grid == 3))  # 1 + maybe for pos

```
## year_2024\day_05\solve_2.py

```py
import sys
from dataclasses import dataclass
from typing import List, Tuple, Set
from numpy.ma.extras import median
from aoc.helpers import locate, build_location, read_lines
from aoc.printer import print_, get_meta_from_fn


sys.setrecursionlimit(3000)

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

```
## year_2024\day_05\solve_1.py

```py
import sys
from typing import List, Tuple
from aoc.helpers import locate, build_location, read_lines
from aoc.printer import get_meta_from_fn, print_


sys.setrecursionlimit(3000)

def is_ordered(ordering_rules, page_updates):
    ok = []
    for i, page_update in enumerate(page_updates):
        must_see = set()
        for j in range(0, len(page_update)):
            mentioned = set(page_update)
            page_number = page_update[j]
            if page_number in must_see:
                must_see.remove(page_number)

            for left, right in ordering_rules:
                if left == page_number:
                    if left in mentioned and right in mentioned:
                        must_see.add(right)
                    continue
        if len(must_see) == 0:
            ok.append(i)

    return ok

def solve_1(__input=None):
    """
    :challenge: 143
    :expect: 6505
    """
    page_ordering_rules: List[Tuple[int]] = []
    page_updates: List[Tuple[int]] = []
    with open(locate(__input), "r") as fp:
        for ind, line in enumerate(read_lines(fp)):
            if len(line) == 0:
                continue
            if "|" in line:
                page_ordering_rules.append(tuple([int(l) for l in line.split("|")]))
            elif line.index(","):
                page_updates.append(tuple([int(l) for l in line.split(",")]))
            else:
                raise TypeError("Can not parse!")

    pristine_page_updates_indices = is_ordered(page_ordering_rules, page_updates)
    return sum(ppu[len(ppu) // 2] for ppu in (page_updates[i] for i in pristine_page_updates_indices))

```
## year_2024\day_04\solve.py

```py
from functools import reduce
from typing import List, Tuple, Any
import numpy as np
from aoc.helpers import locate, build_location, read_lines
from aoc.printer import get_meta_from_fn, print_
from aoc.tools import transpose


def walk_diagonal(s, start_yx: Tuple[int, int], ys, xs):
    word = ""
    for i in range(max(ys, xs)):
        y = i + start_yx[0]
        x = i + start_yx[1]
        if x >= xs:
            break
        if y >= ys:
            break

        word += s[y][x]

    return word

def generate_diagonal(s: List[List[str]] | np.ndarray[Any, np.dtype], ys: int):
    xs = len(s[0])

    for x in range(xs):
        yield walk_diagonal(s, (0, x), ys, xs)

    for y in range(1, ys):
        yield walk_diagonal(s, (y, 0), ys, xs)

def to_word(iterable: List):
    return reduce(lambda r, b: r + b, iterable)

def count(word: str, *patterns: str):
    return sum(word.count(p) for p in patterns)

def solve_1(__input=None):
    """
    :challenge: 18
    :expect: 2575
    """
    input_ = []
    with open(locate(__input), "r") as fp:
        for line in read_lines(fp):
            input_.append(list(line))

    transposed = transpose(input_)
    yx = len(transposed[0])

    total = 0
    for row in transposed:
        word = to_word(row)
        total += count(word, "XMAS", "SAMX")

    for row in input_:
        word = to_word(row)
        total += count(word, "XMAS", "SAMX")

    words = list(generate_diagonal(input_, yx))

    for row in words:
        word = to_word(row)
        total += count(word, "XMAS", "SAMX")

    rotated = np.rot90(input_)
    words = list(generate_diagonal(rotated, yx))

    for row in words:
        word = to_word(row)
        total += count(word, "XMAS", "SAMX")

    return total

def solve_2(__input=None):
    """
    :challenge: 9
    :expect: 2041
    """
    input_ = []
    with open(locate(__input), "r") as fp:
        for line in read_lines(fp):
            input_.append(list(line))

    matrix = np.matrix(input_)
    starting_positions = np.argwhere(matrix == "A")

    xmas = {"MAS", "SAM"}
    total = 0
    for starting_position in starting_positions:
        if 0 in starting_position:
            continue

        y, x = starting_position
        if x == matrix.shape[1] - 1 or y == matrix.shape[0] - 1:
            continue

        word_south_east = f"{matrix[y - 1, x - 1]}{matrix[y, x]}{matrix[y + 1, x + 1]}"
        word_north_east = f"{matrix[y + 1, x - 1]}{matrix[y, x]}{matrix[y - 1, x + 1]}"

        if word_south_east in xmas and word_north_east in xmas:
            total += 1

    return total

```
## year_2024\day_03\solve.py

```py
import re
from aoc.helpers import locate, build_location, read_lines
from aoc.printer import get_meta_from_fn, print_


def solve_1(input_=None):
    """
    :challenge: 161
    :expect: 188741603
    """
    total = 0
    with open(locate(input_), "r") as fp:
        for line in read_lines(fp):
            for a, b in re.findall(r"mul\((\d+),(\d+)\)", line):
                total += int(a) * int(b)

    return total

class MultiplierAutomaton:
    multiply: bool
    total: int

    def __init__(self, start_state: bool):
        self.multiply = start_state
        self.total = 0

    def process(self, cmd: re.Match[str]):
        if cmd[0].startswith("do()"):
            self.multiply = True
        elif cmd[0].startswith("don't()"):
            self.multiply = False
        elif self.multiply:
            self.total += int(cmd[1]) * int(cmd[2])

def solve_2(input_=None):
    """
    :challenge: 48
    :expect: 67269798
    """
    automaton = MultiplierAutomaton(True)
    with open(locate(input_), "r") as fp:
        for line in read_lines(fp):
            for match in re.finditer(r"mul\((\d+),(\d+)\)|don't\(\)|do\(\)", line):
                automaton.process(match)

    return automaton.total

```
## year_2024\day_02\solve.py

```py
import re
from typing import Generator, List, Literal, Set, Tuple
from more_itertools.recipes import sliding_window
from aoc.helpers import build_location, locate, read_lines
from aoc.printer import get_meta_from_fn, print_


def safe(number: Tuple[int] | List[int]):
    ops: Set[Literal["gt", "lt"]] = set()
    for a, b in sliding_window(number, 2):
        if not 0 < abs(a - b) < 4:
            return False
        if a > b:
            ops.add("gt")
        if a < b:
            ops.add("lt")

    return len(ops) == 1

def solve_1(input_=None):
    """
    :challenge: 2
    :expect: 252
    """
    levels = list()
    with open(locate(input_), "r") as fp:
        for line in read_lines(fp):
            levels.append(list(map(int, re.findall(r"\d+", line))))

    return sum(1 for l in levels if safe(l))

def omissions(levels: Tuple[int]) -> Generator[Tuple[int], None, None]:
    for i, _ in enumerate(levels):
        materialized_levels = list(levels)
        materialized_levels.pop(i)
        yield tuple(materialized_levels)

def safe_with_tolerations(level: Tuple[int]):
    return safe(level) or any(1 for o in omissions(level) if safe(o))

def solve_2(input_=None):
    """
    :challenge: 4
    :expect: 324
    """
    levels = list()
    with open(locate(input_), "r") as fp:
        for line in read_lines(fp):
            levels.append(tuple(map(int, re.findall(r"\d+", line))))

    return sum(1 for l in levels if safe_with_tolerations(l))

```
## year_2024\day_01\solve.py

```py
import re
from collections import Counter
from typing import List
from aoc.helpers import build_location, locate, read_lines
from aoc.printer import get_meta_from_fn, print_


def solve_1(input_=None):
    """
    :challenge: 11
    :expect: 1660292
    """
    left, right = [], []

    with open(locate(input_), "r") as fp:
        for line in read_lines(fp):
            l, r = list(map(int, re.findall(r"\d+", line)))
            left.append(l)
            right.append(r)

    left.sort(), right.sort()
    return sum([abs(left[i] - right[i]) for i, _ in enumerate(left)])

def solve_2(input_=None):
    """
    :challenge: 31
    :expect: 22776016
    """
    left: List[int] = []
    right_counter = Counter()

    with open(locate(input_), "r") as fp:
        for line in read_lines(fp):
            [l, r] = list(map(int, re.findall(r"\d+", line)))
            left.append(l)
            right_counter.update({r: 1})

    return sum(left_item * right_counter[left_item] for left_item in left)

```
