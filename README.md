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

## Focus for 2024

- Explore Automatons preferably NFA but also a DFA.
  - Focus transition of states and accepting states.
  - Try to find something for push-down Automata.
  - Stay close to the language used in the puzzles.
- Attempt a CUDA brute at some point.
- Try a pathfinding concept aside A*, Dijkstra, BFS and DFS. Perhaps Bidirectional Search, JPS, D-Lite**, Theta*,
  Bellman-Ford or Floyd-Warshall.


| Type | Algorithm                                         |
|-------|---------------------------------------------------|
|Grid-based games or simulations:| A*, JPS, Theta*                                   |
|Dynamic environments:| D* or [D*-Lite](https://en.wikipedia.org/wiki/D*) |
|Unweighted graphs:| BFS                                               |
|Weighted graphs:| Dijkstra or A*                                    |
|Negative weights:| [Bellman-Ford](https://en.wikipedia.org/wiki/Bellman%E2%80%93Ford_algorithm)                                      |
|Memory constraints:| [IDA*](https://en.wikipedia.org/wiki/Iterative_deepening_A*)                                          |
|All-pairs shortest paths:| [Floyd-Warshall](https://en.wikipedia.org/wiki/Floyd%E2%80%93Warshall_algorithm)                                    |


## year_2024\day_19\solve_2.py

```py
import itertools
import operator
import re
import statistics
import sys
from collections import Counter, OrderedDict, defaultdict
from copy import copy, deepcopy
from dataclasses import dataclass
from functools import reduce
from typing import Dict, List, Callable, Tuple, Literal, Set, Generator, Any
import more_itertools
import numpy as np
from defaultlist import defaultlist
from more_itertools import windowed, chunked
from more_itertools.recipes import sliding_window
from automata.fa.nfa import NFA
from aoc.helpers import locate, build_location, read_lines
from aoc.poll_printer import PollPrinter
from aoc.printer import get_meta_from_fn, print_, ANSIColors, print2
from aoc.tests.test_fixtures import get_challenges_from_meta
from aoc.tools import transpose
from year_2021.day_05 import direction


sys.setrecursionlimit(30_000)

def tuple_sorted(iterable: Set | List | Tuple) -> Tuple:
    return tuple(sorted(iterable))

class TowelNFA:

    def __init__(self, towels):
        assert len(towels) == len(set(towels))
        self.towels = towels
        self.total = 0

    def accepts(self, word):
        i = 0
        total = 0
        current_state = Counter()
        current_state.update({word: 1})
        while 1:
            i += 1
            canonical_state = dict(current_state)
            current_state.clear()

            for state, count in canonical_state.items():
                matches = Counter()

                if state == "":
                    total += count
                    continue

                for towel in self.towels:
                    if state.startswith(towel):
                        new_state = state[len(towel):]
                        matches.update({(state, new_state): count})

                # Reduce "from state" with count.
                for k, q in matches.items():
                    current_state.update({k[1]: q})

            if set(canonical_state.keys()) == set(current_state.keys()):
                break

        # print(f"word: {word} len: {total}")
        self.total += total

        return self.total

def solve_(__input=None):
    """
    :challenge: 16
    :expect: 1100663950563322
    """
    words = []
    with open(locate(__input), "r") as fp:
        lines = read_lines(fp)
        towels = lines[0].split(", ")
        for word in lines[1:]:
            words.append(word)

    nfa = TowelNFA(towels)

    n = 0
    for word in words:
        if nfa.accepts(word):
            n += 1

    return nfa.total

```
## year_2024\day_19\solve_1.py

```py
import itertools
import operator
import re
import statistics
import sys
from collections import Counter, OrderedDict, defaultdict
from copy import copy, deepcopy
from dataclasses import dataclass
from functools import reduce
from typing import Dict, List, Callable, Tuple, Literal, Set, Generator, Any
import more_itertools
import numpy as np
from defaultlist import defaultlist
from more_itertools import windowed, chunked
from more_itertools.recipes import sliding_window
from automata.fa.nfa import NFA
from aoc.helpers import locate, build_location, read_lines
from aoc.poll_printer import PollPrinter
from aoc.printer import get_meta_from_fn, print_, ANSIColors, print2
from aoc.tests.test_fixtures import get_challenges_from_meta
from aoc.tools import transpose
from year_2021.day_05 import direction


sys.setrecursionlimit(30_000)

class TowelAutomaton:

    def __init__(self, towels):
        self.towels = towels

    def accepts(self, word):
        last_states = {word}
        while 1:
            canonical_state = set(last_states)
            for towel in self.towels:
                current_state = set()
                for state in last_states:
                    if state == "":
                        return True

                    current_state.add(state)
                    if state.startswith(towel):
                        current_state.add(state[len(towel):])

                last_states = current_state

            if canonical_state == last_states:
                break

        return False

def solve_(__input=None):
    """
    :challenge: 6
    :expect: 374
    """
    words = []
    with open(locate(__input), "r") as fp:
        lines = read_lines(fp)
        towels = lines[0].split(", ")
        for word in lines[1:]:
            words.append(word)

    nfa = TowelAutomaton(towels)

    n = 0
    total = list()
    for word in words:
        if nfa.accepts(word):
            total.append(word)
            n += 1

    return sum(1 for _ in total)

```
## year_2024\day_18\solve_2.py

```py
import heapq
import itertools
import operator
import re
import statistics
import sys
import traceback
from bisect import bisect
from collections import Counter, OrderedDict, defaultdict
from copy import copy, deepcopy
from dataclasses import dataclass
from functools import reduce
from typing import Dict, List, Callable, Tuple, Literal, Set, Generator, Any
import more_itertools
import numpy as np
from defaultlist import defaultlist
from more_itertools import windowed, chunked
from more_itertools.recipes import sliding_window
from aoc.helpers import locate, build_location, read_lines
from aoc.poll_printer import PollPrinter
from aoc.printer import get_meta_from_fn, print_, ANSIColors, print2
from aoc.tests.test_fixtures import get_challenges_from_meta
from aoc.tools import transpose, pretty
from year_2021.day_05 import direction


sys.setrecursionlimit(30_000)

def get_neighbors(node, grid):
    von_neumann = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    neighbors = []

    for dy, dx in von_neumann:
        neighbor = node[0] + dy, node[1] + dx

        if -1 < neighbor[0] < grid.shape[0] and -1 < neighbor[1] < grid.shape[1]:
            if grid[neighbor] != "#":
                neighbors.append(neighbor)

    return neighbors

def dijkstra(grid, start, end, costs):
    seen = set()
    q = [start]
    costs[start] = 0
    # heapq.heapify(pq)
    while q:
        current = q.pop(0)

        if current == end:
            break

        if current in seen:
            continue

        seen.add(current)

        for n in get_neighbors(current, grid):
            neighbor_cost = costs[n]
            current_cost = costs[current]
            if neighbor_cost > current_cost + 1:
                costs[n] = current_cost + 1
                if n in seen:
                    seen.remove(n)

            q.append(n)

def run_dijkstra(coords, bytes_: int, shape, costs):
    grid = np.full(shape, dtype=str, fill_value=".")
    for y, x in coords[:bytes_]:
        grid[y, x] = "#"

    start = (0, 0)
    end = (shape[0] - 1, shape[1] - 1)
    dijkstra(grid, start, end, costs)

def solve_(__input=None):
    """
    :challenge: 6,1
    :expect: 26,50
    """
    coords = []
    with open(locate(__input), "r") as fp:
        for line in read_lines(fp):
            x, y = line.split(",")
            coords.append((int(y), int(x)))

    if "test" in  __input:
        shape = (7, 7)
        bytes_ = 12
    else:
        shape = (71, 71)
        bytes_ = 1024

    max_bytes = len(coords)

    start = (0, 0)
    end = (shape[0] - 1, shape[1] - 1)

    # Classic bisect w/o built-in Python variant. Learn this through.
    min_ = bytes_
    max_ = max_bytes
    while 1:
        costs = np.full(shape, dtype=int, fill_value=sys.maxsize)
        delta = max_ - min_

        if delta < 2:
            y, x = coords[min_]
            return f"{x},{y}"

        bisect_at = min_ + delta // 2
        run_dijkstra(coords, bisect_at, shape, costs)

        if costs[end] == sys.maxsize:
            max_ = bisect_at
        else:
            min_ = bisect_at

```
## year_2024\day_18\solve_1.py

```py
import heapq
import itertools
import operator
import re
import statistics
import sys
import traceback
from collections import Counter, OrderedDict, defaultdict
from copy import copy, deepcopy
from dataclasses import dataclass
from functools import reduce
from typing import Dict, List, Callable, Tuple, Literal, Set, Generator, Any
import more_itertools
import numpy as np
from defaultlist import defaultlist
from more_itertools import windowed, chunked
from more_itertools.recipes import sliding_window
from aoc.helpers import locate, build_location, read_lines
from aoc.poll_printer import PollPrinter
from aoc.printer import get_meta_from_fn, print_, ANSIColors, print2
from aoc.tests.test_fixtures import get_challenges_from_meta
from aoc.tools import transpose, pretty
from year_2021.day_05 import direction


sys.setrecursionlimit(30_000)

def get_neighbors(node, grid):
    von_neumann = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    neighbors = []

    for dy, dx in von_neumann:
        neighbor = node[0] + dy, node[1] + dx

        if -1 < neighbor[0] < grid.shape[0] and -1 < neighbor[1] < grid.shape[1]:
            if grid[neighbor] != "#":
                neighbors.append(neighbor)

    return neighbors

def dijkstra(grid, start, end, costs):
    seen = set()
    q = [start]
    costs[start] = 0
    # heapq.heapify(pq)
    while q:
        current = q.pop(0)

        if current == end:
            break

        if current in seen:
            continue

        seen.add(current)

        for n in get_neighbors(current, grid):
            neighbor_cost = costs[n]
            current_cost = costs[current]
            if neighbor_cost > current_cost + 1:
                costs[n] = current_cost + 1
                if n in seen:
                    seen.remove(n)

            q.append(n)
            # heapq.heappush(pq, n)

def solve_(__input=None):
    """
    :challenge: 22
    :expect: 380
    """
    coords = []
    with open(locate(__input), "r") as fp:
        for line in read_lines(fp):
            x, y = line.split(",")
            coords.append((int(y), int(x)))

    if "test" in  __input:
        shape = (7, 7)
        bytes_ = 12
    else:
        shape = (71, 71)
        bytes_ = 1024

    grid = np.full(shape, dtype=str, fill_value=".")
    costs = np.full(grid.shape, dtype=int, fill_value=sys.maxsize)

    for y, x in coords[:bytes_]:
        grid[y, x] = "#"

    start = (0, 0)
    end = (shape[0] - 1, shape[1] - 1)

    dijkstra(grid, start, end, costs)

    return costs[end]

```
## year_2024\day_17\solve_1.py

```py
import re
import sys
from aoc.helpers import locate, build_location, read_lines
from aoc.printer import get_meta_from_fn, ANSIColors, print2
from year_2024.day_17.chronospatial_computer import ChronospatialComputer


sys.setrecursionlimit(30_000)

def solve_(__input=None):
    # challenge 4,6,3,5,6,3,5,2,1,0.
    # expect 2,1,3,0,5,2,3,7,1
    """
    :challenge_2: -1
    :expect: -1
    """
    lines = []
    with open(locate(__input), "r") as fp:
        for line in read_lines(fp):
            lines.append(line)

    a, b, c = [int(m) for m in re.findall(r"\d+", "".join(lines[:3]))]
    optcodes = [int(m) for m in re.findall(r"\d+", lines[3])]

    outcome = []
    computer = ChronospatialComputer(a, b, c, optcodes, outcome.append)

    while optcode := computer.next():
        computer.set_operation(optcode[0])
        computer.set_operand(optcode[1])
        computer.apply()

    return ",".join(str(o) for o in outcome)

```
## year_2024\day_16\solve_1.py

```py
import heapq
import itertools
import sys
from enum import Enum
from typing import List, Tuple, Generator, Self
import numpy as np
from aoc.helpers import locate, build_location, read_lines
from aoc.printer import get_meta_from_fn, ANSIColors, print2


sys.setrecursionlimit(30_000)

class Heading(Enum):
    N = 0
    E = 1
    S = 2
    W = 3

    def von_neumann(self) -> Tuple[int, int]:
        match self:
            case Heading.N: return -1, 0
            case Heading.S: return 1, 0
            case Heading.W: return 0, -1
            case Heading.E: return 0, 1

    def make_turns(self) -> List[Self]:
        if self == Heading.N or self == Heading.S:
            return [Heading.W.value, Heading.E.value]
        return [Heading.N.value, Heading.S.value]

def neighbors_and_cost_delta(heading: int, y: int, x: int, grid) -> Generator[Tuple[Heading, int, int, int], None, None]:
    heading_explorer = Heading(heading)
    for turned in heading_explorer.make_turns():
        yield turned, y, x, 1000

    dy, dx = heading_explorer.von_neumann() # a move
    new_node = y + dy, x + dx

    cell_value = str(grid[new_node])
    if cell_value != "#":
        yield heading, new_node[0], new_node[1], 1

def heuristics_manhattan(a: Tuple[int, int], b: Tuple[int, int]):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def dijkstra(begin_pos: Tuple[int, int], begin_heading: Heading, end: Tuple[int, int], costs, grid):
    """
    Dijkstra-variant with PQ and heuristics for cut-off and priority. Not quite A* either.
    """
    begin = (begin_heading.value, *begin_pos)
    pq: List[Tuple[int, int, int, int]] = [(0, *begin)]
    costs[begin_heading.value, begin[1], begin[2]] = 0
    seen = set()
    while pq:
        _, heading, y, x, = heapq.heappop(pq)
        current = heading, y, x
        current_cost = costs[heading, y, x]

        if current in seen:
            continue

        leads = list(neighbors_and_cost_delta(heading, y, x, grid))
        for neighbor_heading, ny, nx, delta_cost in leads:
            previous_cost = costs[neighbor_heading, ny, nx]
            total_cost = current_cost + delta_cost
            neighbor = neighbor_heading, ny, nx
            if previous_cost > total_cost:
                costs[neighbor] = total_cost
                if neighbor in seen:
                    seen.remove(neighbor)

            h = heuristics_manhattan((ny, nx), end)
            heapq.heappush(pq, (h, neighbor_heading, ny, nx))

        seen.add(current)

def solve_(__input=None):
    """
    :challenge: 7036
    :challenge_2: 11048
    :expect: 73432
    """
    lines = []
    with open(locate(__input), "r") as fp:
        for line in read_lines(fp):
            lines.append(list(line))

    grid = np.matrix(lines, dtype=str)
    cost_shape = (4, *grid.shape)
    costs = np.full(cost_shape, dtype=np.int64, fill_value=sys.maxsize)
    begin, = [(int(y), int(x)) for y, x in np.argwhere(grid == 'S')]
    end, = [(int(y), int(x)) for y, x in np.argwhere(grid == 'E')]
    begin_heading = Heading.E

    dijkstra(begin, begin_heading, end, costs, grid)

    # For troubleshooting
    cost_summary = np.zeros(grid.shape)

    for z, y, x in itertools.product(range(cost_shape[0]), range(cost_shape[1]), range(cost_shape[2])):
        if (cost_summary[y, x] == 0 or costs[z, y, x] < cost_summary[y, x]) and costs[z, y, x] != sys.maxsize:
            cost_summary[y, x] = costs[z, y, x]

    return costs[:,end[0],end[1]].min()

```
## year_2024\day_15\solve_2.py

```py
import itertools
import re
import sys
from typing import Dict, List, Tuple
import numpy as np
from aoc.helpers import locate, build_location, read_lines
from aoc.printer import get_meta_from_fn, ANSIColors, print2


sys.setrecursionlimit(30_000)

class StorageRobotAutomaton:
    lookup: Dict[str, Tuple[int, int]]

    def __init__(self, grid):
        self.grid = grid
        self.lookup = { "^": (-1, 0), "v": (1, 0), ">": (0, 1), "<": (0, -1) }
        self.sideways_peek_lookup = { "[": (0, 1), "]": (0, -1) }

    def _peek(self, direction_: str, y0: int, x0: int) -> List[Tuple[int, int]] | str:
        # Peek(y0, x0, self.grid[y0, x0])
        dy0, dx0 = self.lookup[direction_]
        y1, x1 = dy0 + y0, dx0 + x0
        cell_value = self.grid[y1, x1]
        if more := self.sideways_peek_lookup.get(cell_value):
            dy1, dx1 = more
            return [(y1, x1), (y1 + dy1, x1 + dx1)]

        return cell_value

    def peek(self, direction_: str, y0: int, x0: int) -> List[Tuple[int, int]]:
        heap, ok, added = [(y0, x0)], [], set()

        while heap:
            y, x = heap.pop()
            if (peeked := self._peek(direction_, y, x)) == ".":
                ok.append((y, x))
                continue
            elif peeked == "#":
                return []

            if isinstance(peeked, List):
                ok.append((y, x))
                for p in (p for p in peeked if p not in added):
                    heap.append(p)
                    added.add(p)
        return ok

    def shove(self, direction_: str, n: int) -> None:
        # before_grid = deepcopy(self.grid)
        robot, = np.argwhere(self.grid == "@").tolist()
        ok = self.peek(direction_, *robot)
        seen = set()
        assigned = set()
        if len(ok) > 0:
            values = [(*p, self.grid[*p]) for p in ok]
            for y0, x0, value in values:
                seen.add((y0, x0))
                dy0, dx0 = self.lookup[direction_]
                yx0 = y0 + dy0, x0 + dx0
                self.grid[yx0] = value
                assigned.add(yx0)

            unassigned = [s for s in seen if s not in assigned]
            for u in unassigned:
                self.grid[u] = "."

        # print(f"Move {direction_} ({n}):\r\nbefore:\r\r{pretty(before_grid)}\r\n after:\r\n{pretty(self.grid)}\r\n")

    def sum_goods_positioning_system(self):
        blocks = np.argwhere(self.grid == "[")
        return sum(100 * y + x for y, x in blocks)

def solve_(__input=None):
    """
    :challenge: 9021
    :expect: 1550677
    """
    _instructions = ""
    _grid = []
    with open(locate(__input), "r") as fp:
        for line in read_lines(fp):
            if re.search(r'[<v^>]', line):
                _instructions += line
            else:
                _grid.append(list(line))

    programming = list(_instructions)

    shape = len(_grid), len(_grid[0])
    new_shape = shape[0], shape[1] * 2
    grid = np.full(new_shape, dtype=str, fill_value=".")
    for y, x in itertools.product(range(shape[0]), range(shape[1])):
        match _grid[y][x]:
            case ".":
                pass
            case "@":
                grid[y, x * 2] = "@"
            case "O":
                grid[y, x * 2] = "["
                grid[y, x * 2 + 1] = "]"
            case "#":
                grid[y, x * 2] = "#"
                grid[y, x * 2 + 1] = "#"

    storage_robot = StorageRobotAutomaton(grid)
    for n, move_direction in enumerate(programming):
        storage_robot.shove(move_direction, n)

    return storage_robot.sum_goods_positioning_system()

```
## year_2024\day_15\solve_1.py

```py
import re
import sys
import numpy as np
from aoc.helpers import locate, build_location, read_lines
from aoc.printer import get_meta_from_fn, ANSIColors, print2


sys.setrecursionlimit(30_000)

class StorageRobotAutomaton:

    def __init__(self, grid):
        self.grid = grid

    def assign(self, ys: slice, xs: slice, array, __dir: str):
        i = 0
        for y in range(ys.start, ys.stop):
            for x in range(xs.start, xs.stop):
                self.grid[y, x] = array[i]
                i += 1

    def shove(self, dir_: str, ) -> bool:
        robot, = np.argwhere(self.grid == "@")
        y, x = robot
        reverse = False
        accumulated = ["@"]
        found = False
        pos = [y, x]
        while not found and - 1 < pos[0] < self.grid.shape[0] and - 1 < pos[1] < self.grid.shape[1]:
            match dir_:
                case '^':
                    pos[0] -= 1
                    reverse = True
                case 'v':
                    pos[0] += 1
                case '<':
                    pos[1] -= 1
                    reverse = True
                case '>':
                    pos[1] += 1
                case _:
                    raise NotImplemented("What!")

            value = self.grid[*pos]
            if value == ".":  #  and accumulated:
                y0 = sorted([pos[0], y])
                y0[-1] = y0[-1] + 1
                x0 = sorted([pos[1], x])
                x0[-1] = x0[-1] + 1
                accumulated.insert(0, ".")
                if reverse:
                    accumulated.reverse()

                self.assign(slice(*y0), slice(*x0), accumulated, dir_)
                return True
            elif value == "#":
                return False
            else:
                accumulated.append(value)

    def sum_goods_positioning_system(self):
        blocks = np.argwhere(self.grid == "O")
        return sum(100 * y + x for y, x in blocks)

def solve_(__input=None):
    """
    :challenge: 2028
    :challenge_2: 10092
    :expect: 1526018
    """
    _programming = ""
    _maze = []
    with open(locate(__input), "r") as fp:
        for line in read_lines(fp):
            if re.search(r'[<v^>]', line):
                _programming += line
            else:
                _maze.append(list(line))

    programming = list(_programming)
    grid = np.matrix(_maze, dtype=str)

    storage_robot = StorageRobotAutomaton(grid)
    for direction in programming:
        storage_robot.shove(direction)

    return storage_robot.sum_goods_positioning_system()

```
## year_2024\day_14\solve_2.py

```py
import heapq
import operator
import re
import sys
from typing import List, Tuple, Any, Optional
import numpy as np
from aoc.helpers import locate, build_location, read_lines
from aoc.printer import get_meta_from_fn, ANSIColors, print2


sys.setrecursionlimit(30_000)

def move(shape: Tuple[int, int], robots: List, turns: int, print: bool = False):
    k = turns
    moved_robots = []
    for n, robot in enumerate(robots):
        p0, p1, v0, v1 = robot
        # p0, p1, k, v0, v1, height, width = symbols("p0 p1 k v0 v1 height width")

        y = (p1 + k * v1) % shape[0]
        x = (p0 + k * v0) % shape[1]
        moved_robots.append((n, y, x))

    return moved_robots

def even_shape(shape: Tuple[int, int]) -> Tuple[int, int]:
    y = shape[0] if shape[0] % 2 == 0 else shape[0] + 1
    x = shape[1] if shape[1] % 2 == 0 else shape[1] + 1
    return y, x

Candidates = np.ndarray[Any, np.dtype[np.unsignedinteger[np.uint8]]]

def quantize(shape: Tuple[int, int], every_robot: Candidates, z:int, slices: Optional[Tuple[slice, slice]], min_size=3) -> Tuple[int, int, int]:
    y, x = slices if slices else (slice(0, 0), slice(0, 0))
    for dy, dx in ((0, 0), (1, 0), (0, 1), (1, 1)):
        _y = slice(y.start + shape[0] * dy // 2, y.start + shape[0] * (dy + 1) // 2)
        _x = slice(x.start + shape[1] * dx // 2, x.start + shape[1] * (dx + 1) // 2)
        if _y.stop - _y.start < min_size or _x.stop - _x.start < min_size:
            continue
        yield int(every_robot[z, _y, _x].sum()), _y, _x

def every_robot_position_as_3d_array(grid_shape: Tuple[int, int], actual_shape: Tuple[int, int], repeat_at: int, robots: List[List[int]]) -> Candidates:
    """
    Initializes an array a bit greedily at 20_000 at z-index. Alternate approaches:
    # 1. Symbolic math (sympy) for when all robots end up starting position again (k)=
    # 2. GCD
    # 3. hash
    """
    candidates = np.zeros((20_000, *grid_shape), dtype=np.uint8)
    for z in range(1, repeat_at):
        moved_robots = move(actual_shape, robots, z)
        for name, y, x in moved_robots:
            candidates[z, y, x] = 1

    return candidates

def create_priority_queue(repeat_at: int, grid_shape: Tuple[int, int], candidates: Candidates):
    heap = []
    heapq.heapify(heap)
    for z in range(1, repeat_at):
        for sum_, _y, _x in quantize(grid_shape, candidates, z, None):
            potential = (_y.stop - _y.start) * (_x.stop - _x.start)
            heapq.heappush(heap, (float(-sum_) / potential, sum_, z, _y, _x))
    return heap

def solve_(__input=None):
    """
    :expect: 7338
    """
    robots = []
    with open(locate(__input), "r") as fp:
        for line in read_lines(fp):
            robot = list(map(int, re.findall(r"-?\d+", line)))
            robots.append(robot)

    shape = (7, 11) if "test" in __input else (103, 101)
    grid_shape = even_shape(shape)

    repeat_at = 10403
    every_robot_position = every_robot_position_as_3d_array(grid_shape, shape, repeat_at, robots)
    heap = create_priority_queue(repeat_at, grid_shape, every_robot_position)

    while heap:
        _, sum_, z, y, x = heapq.heappop(heap)
        shape = y.stop - y.start, x.stop - x.start

        if sum_ == (y.stop - y.start) * (x.stop - x.start):  # every value in quant has robot
            return z

        for sum_, _y, _x in quantize(shape, every_robot_position, z, (y, x)):
            _shape = _y.stop - _y.start, _x.stop - _x.start
            _potential = operator.mul(*_shape)
            heapq.heappush(heap, (-sum_ / _potential, sum_, z, _y, _x))

```
## year_2024\day_14\solve_1.py

```py
import math
import math
import operator
import re
import sys
from collections import Counter
from copy import deepcopy
from functools import reduce
from typing import List, Tuple, Generator
import numpy as np
from aoc.helpers import locate, build_location, read_lines
from aoc.printer import get_meta_from_fn, ANSIColors, print2
from aoc.tools import pretty


sys.setrecursionlimit(30_000)

def move(shape: Tuple[int, int], robots: List, turns: int, print: bool = False):
    k = turns
    moved_robots = []
    for n, robot in enumerate(robots):
        p0, p1, v0, v1 = robot
        # p0, p1, k, v0, v1, height, width = symbols("p0 p1 k v0 v1 height width")

        y = (p1 + k * v1) % shape[0]
        x = (p0 + k * v0) % shape[1]
        moved_robots.append((n, y, x))

    return moved_robots

def each_tick_print(shape, robots: list, ticks = 5):
    grid_ = np.zeros(shape, dtype=np.int32)

    for n, robot in enumerate(robots):
        p0, p1, v0, v1 = robot
        grid_[p1, p0] = grid_[p1, p0] + 1

    print("Initial state:\n" + pretty(grid_) + "\n")

    for tick in range(1, ticks + 1):
        moved_robots_1 = move(shape, robots, tick)
        plural = "s" if tick > 1 else ""

        grid_1 = np.zeros(shape, dtype=np.int32)

        for n, robot in enumerate(moved_robots_1):
            n, y, x = robot
            grid_1[y, x] = grid_1[y, x] + 1

        print(f"After {tick} second{plural}:\n" + pretty(grid_1) + "\n")

def solve_(__input=None):
    """
    :challenge: 12
    :expect: 223020000
    """
    lines = []
    with open(locate(__input), "r") as fp:
        for line in read_lines(fp):
            lines.append(line)

    robots = []
    for line in lines:
        robot = list(map(int, re.findall(r"-?\d+", line)))
        robots.append(robot)

    shape = (7, 11) if "test" in __input else (103, 101)
    troubleshoot = True if "test_2" in __input else False

    each_tick_print(shape, robots)

    moved_robots = move(shape, robots, 100, troubleshoot)

    _grid = np.zeros(shape, dtype=np.int32)

    totals = Counter()
    for name, y, x in moved_robots:
        if y - shape[0] // 2 == 0:
            continue

        if x - shape[1] // 2 == 0:
            continue

        _grid[y, x] = 1 + _grid[y, x]

        qy = y // ((shape[0] // 2) + 1)
        qx = x // ((shape[1] // 2) + 1)
        totals.update({(qy, qx): 1})

    print(pretty(_grid))

    return reduce(operator.mul, totals.values())

```
## year_2024\day_13\solve_2.py

```py
import math
import re
import sys
from math import gcd
import more_itertools
from sympy import symbols, Eq
from sympy.solvers import solve
from aoc.helpers import locate, build_location, read_lines
from aoc.printer import get_meta_from_fn, ANSIColors, print2
from year_2024.day_13.solve_1 import is_integer


sys.setrecursionlimit(30_000)

def _solve(_input=None):
    """
    :challenge: 0
    :expect: 76358113886726
    """
    lines = []
    with open(locate(_input), "r") as fp:
        for line in read_lines(fp):
            lines.append(line)

    machines = []
    for button_a, button_b, prize_ in more_itertools.chunked(lines, 3):
        assert "Button A" in button_a
        assert "Button B" in button_b
        assert "Prize" in prize_
        a = map(int, re.findall(r"\d+", button_a))
        b = map(int, re.findall(r"\d+", button_b))
        prize = map(int, re.findall(r"\d+", prize_))
        machines.append((a, b, prize))

    total = 0
    for n, opt_machine in enumerate(machines):
        _ax, _ay = opt_machine[0]
        _bx, _by = opt_machine[1]
        _px, _py = opt_machine[2]
        a, b, ax, bx, ay, by, px, py = symbols("a b ax bx ay by px py")

        eq0 = Eq(a * ax + b * bx, _px)
        eq1 = Eq(a * ay + b * by, _py)
        eq2 = eq0.subs({ax: _ax, bx: _bx})
        eq3 = eq1.subs({ay: _ay, by: _by})
        solve0 = solve([eq2, eq3], (a, b))
        if is_integer(solve0[a]) and is_integer(solve0[b]):
            total += solve0[a] * 3 + solve0[b]

    return total

```
## year_2024\day_13\solve_1.py

```py
import math
import re
import sys
from math import gcd
import more_itertools
from sympy import symbols, Eq
from sympy.solvers import solve
from aoc.helpers import locate, build_location, read_lines
from aoc.printer import get_meta_from_fn, ANSIColors, print2


sys.setrecursionlimit(30_000)

def is_integer(value) -> bool:
    return math.floor(value) == value

def _solve(__input=None):
    """
    :challenge: 480
    :expect: 36758
    """
    lines = []
    with open(locate(__input), "r") as fp:
        for line in read_lines(fp):
            lines.append(line)

    machines = []
    for button_a, button_b, prize_ in more_itertools.chunked(lines, 3):
        assert "Button A" in button_a
        assert "Button B" in button_b
        assert "Prize" in prize_
        a = map(int, re.findall(r"\d+", button_a))
        b = map(int, re.findall(r"\d+", button_b))
        prize = map(int, re.findall(r"\d+", prize_))
        machines.append((a, b, prize))

    total = 0
    for n, opt_machine in enumerate(machines):
        _ax, _ay = opt_machine[0]
        _bx, _by = opt_machine[1]
        _px, _py = opt_machine[2]
        a, b, ax, bx, ay, by, px, py = symbols("a b ax bx ay by px py")

        eq0 = Eq(a * ax + b * bx, _px)
        eq1 = Eq(a * ay + b * by, _py)
        eq2 = eq0.subs({ax: _ax, bx: _bx})
        eq3 = eq1.subs({ay: _ay, by: _by})
        solve0 = solve([eq2, eq3], (a, b))
        if is_integer(solve0[a]) and is_integer(solve0[b]):
            total += solve0[a] * 3 + solve0[b]

    return total

```
## year_2024\day_12\solve_2.py

```py
import itertools
import operator
import re
import statistics
import sys
from collections import Counter, OrderedDict, defaultdict
from copy import copy, deepcopy
from dataclasses import dataclass
from enum import Enum
from functools import reduce
from itertools import combinations
from typing import Dict, List, Callable, Tuple, Literal, Set, Generator, Any
import more_itertools
import numpy as np
from defaultlist import defaultlist
from more_itertools import windowed, chunked
from more_itertools.recipes import sliding_window
from numpy._core._multiarray_umath import StringDType
from numpy.random import permutation
from aoc.helpers import locate, build_location, read_lines
from aoc.poll_printer import PollPrinter
from aoc.printer import get_meta_from_fn, print_, ANSIColors, print2
from aoc.tests.test_fixtures import get_challenges_from_meta
from aoc.tools import transpose, group_by
from year_2021.day_05 import direction


sys.setrecursionlimit(30_000)

class FloodFillAutomaton:
    visited: Set[Tuple[int, int]]
    counter: Counter

    def __init__(self, grid):
        self.visited = set()
        self.grid = grid
        self.new_grid = grid = np.full(grid.shape, dtype=StringDType, fill_value=".")
        self.counter = Counter()

    def flood(self, pos: Tuple[int, int]):
        cell = self.grid[pos]
        n = self.counter.get(self.grid[pos], 0)
        name = f"{cell}{n}"
        queue = {pos}
        added = False
        while queue:
            current = queue.pop()

            if current in self.visited:
                continue

            value = None
            neighbors = [(1, 0), (-1, 0), (0, 1), (0, -1)]
            for dy, dx in neighbors:
                y = current[0] - dy
                x = current[1] - dx
                if (y, x) not in self.visited:
                    if self.grid[(y, x)] == ".":
                        continue

                    if self.grid[(y, x)] == self.grid[pos]:
                        queue.add((y, x))

            n = self.counter.get(self.grid[current], 0)
            s = f"{self.grid[current]}{n}"
            added = True
            self.new_grid[current] = value or s
            self.visited.add(current)

        if added:
            self.counter.update({cell: 1})

    def get_grid(self):
        return self.new_grid

def _solve(__input=None):
    """
    :challenge: 80
    :expect: 881182
    """
    _lines = []
    with open(locate(__input), "r") as fp:
        for __line in read_lines(fp):
            _lines.append(list(__line))

    height = len(_lines) + 2
    width = len(_lines[0]) + 2
    shape = height, width
    _grid = np.full(shape, fill_value='.', dtype=str)

    # print(grid)

    for y in range(len(_lines)):
        for x in range(len(_lines[0])):
            _grid[y + 1, x + 1] = _lines[y][x]

    fill = FloodFillAutomaton(_grid)
    positions = ((int(y), int(x)) for y, x in np.argwhere(_grid != "."))
    for pos in positions:
        fill.flood(pos)

    grid = fill.get_grid()

    faces = defaultdict(set)
    for y1, y2 in sliding_window(range(shape[0]), 2):
        for x in range(shape[1]):
            if len((faces_ := {grid[y1, x], grid[y2, x]})) > 1:
                for face in filter(lambda f: f != ".", faces_):
                    association = 'up' if grid[y1, x] == face else 'down'
                    faces[face].add((y1, y2, x, 'y', association))

    for x1, x2 in sliding_window(range(shape[0]), 2):
        for y in range(shape[0]):
            if len((faces_ := {grid[y, x1], grid[y, x2]})) > 1:
                for face in filter(lambda f: f != ".", faces_):
                    association = 'left' if grid[y, x1] == face else 'right'
                    faces[face].add((x1, x2, y, 'x', association))

    for name, faces_ in faces.items():
        mat = list(filter(lambda r: r[3] == "y", faces_))
        iterable = sorted(mat, key=lambda t: (t[0], t[1], t[2]))
        sw = list(more_itertools.sliding_window(iterable, 2))
        for a, b in sw:
            # if (a[0], a[1]) == (b[0], b[1]) and abs(a[2] - b[2]) < 2 and (grid[a[0], a[2]], grid[a[1], a[2]]) == (grid[b[0], b[2]], grid[b[1], b[2]]):
            one = a[-1]
            two = b[-1]
            if (a[0], a[1]) == (b[0], b[1]) and abs(a[2] - b[2]) < 2 and one == two:
                faces[name].remove(a)

    for name, faces_ in faces.items():
        mat = list(filter(lambda r: r[3] == "x", faces_))
        iterable = sorted(mat, key=lambda t: (t[0], t[1], t[2]))
        sw = list(more_itertools.sliding_window(iterable, 2))
        for a, b in sw:
            # if (a[0], a[1]) == (b[0], b[1]) and abs(a[2] - b[2]) < 2 and (grid[a[0], a[2]], grid[a[1], a[2]]) == (grid[b[0], b[2]], grid[b[1], b[2]]):
            if (a[0], a[1]) == (b[0], b[1]) and abs(a[2] - b[2]) < 2 and a[-1] == b[-1]:
                faces[name].remove(a)

    unique, counts = np.unique(grid, return_counts=True)
    areas = dict(zip(unique, counts))

    total = 0
    for k in sorted(unique):
        if k == '.':
            continue
        area = areas[k]
        perimeter = len(faces[k])
        k_local = perimeter * area
        # print(f"{k} = {k_local}")
        total += k_local

    return total

```
## year_2024\day_12\solve_1.py

```py
import itertools
import operator
import re
import statistics
import sys
from collections import Counter, OrderedDict, defaultdict
from copy import copy, deepcopy
from dataclasses import dataclass
from enum import Enum
from functools import reduce
from itertools import combinations
from typing import Dict, List, Callable, Tuple, Literal, Set, Generator, Any
import more_itertools
import numpy as np
from defaultlist import defaultlist
from more_itertools import windowed, chunked
from more_itertools.recipes import sliding_window
from numpy._core._multiarray_umath import StringDType
from numpy.random import permutation
from aoc.helpers import locate, build_location, read_lines
from aoc.poll_printer import PollPrinter
from aoc.printer import get_meta_from_fn, print_, ANSIColors, print2
from aoc.tests.test_fixtures import get_challenges_from_meta
from aoc.tools import transpose
from year_2021.day_05 import direction


sys.setrecursionlimit(30_000)

class FloodFillAutomaton:
    visited: Set[Tuple[int, int]]
    counter: Counter

    def __init__(self, grid):
        self.visited = set()
        self.grid = grid
        self.new_grid = grid = np.full(grid.shape, dtype=StringDType, fill_value=".")
        self.counter = Counter()

    def flood(self, pos: Tuple[int, int]):
        cell = self.grid[pos]
        n = self.counter.get(self.grid[pos], 0)
        name = f"{cell}{n}"
        queue = {pos}
        added = False
        while queue:
            current = queue.pop()

            if current in self.visited:
                continue

            value = None
            neighbors = [(1, 0), (-1, 0), (0, 1), (0, -1)]
            for dy, dx in neighbors:
                y = current[0] - dy
                x = current[1] - dx
                if (y, x) not in self.visited:
                    if self.grid[(y, x)] == ".":
                        continue

                    if self.grid[(y, x)] == self.grid[pos]:
                        queue.add((y, x))

            n = self.counter.get(self.grid[current], 0)
            s = f"{self.grid[current]}{n}"
            added = True
            self.new_grid[current] = value or s
            self.visited.add(current)

        if added:
            self.counter.update({cell: 1})

    def get_grid(self):
        return self.new_grid

def solve(__input=None):
    """
    :challenge: 140
    :expect: 1467094
    """
    _lines = []
    with open(locate(__input), "r") as fp:
        for __line in read_lines(fp):
            _lines.append(list(__line))

    ylen = len(_lines) + 2
    xlen = len(_lines[0]) + 2
    shape = ylen, xlen
    _grid = np.full(shape, fill_value='.', dtype=str)

    # print(grid)

    for y in range(len(_lines)):
        for x in range(len(_lines[0])):
            _grid[y + 1, x + 1] = _lines[y][x]

    fill = FloodFillAutomaton(_grid)
    positions = ((int(y), int(x)) for y, x in np.argwhere(_grid != "."))
    for pos in positions:
        fill.flood(pos)

    grid = fill.get_grid()

    faces = defaultdict(set)
    for y1, y2 in sliding_window(range(shape[0]), 2):
        for x in range(shape[1]):
            if len((faces_ := {grid[y1, x], grid[y2, x]})) > 1:
                for face in filter(lambda f: f != ".", faces_):
                    faces[face].add((y1, y2, x, 'y'))

    for x1, x2 in sliding_window(range(shape[1]), 2):
        for y in range(shape[0]):
            if len((faces_ := {grid[y, x1], grid[y, x2]})) > 1:
                for face in filter(lambda f: f != ".", faces_):
                    faces[face].add((x1, x2, y, 'x'))

    unique, counts = np.unique(grid, return_counts=True)
    areas = dict(zip(unique, counts))

    total = 0
    for k in sorted(unique):
        if k == '.':
            continue
        area = areas[k]
        perimeter = len(faces[k])
        k_local = perimeter * area
        # print(f"{k} = {k_local}")
        total += k_local

    return total

```
## year_2024\day_11\solve_2.py

```py
import functools
import sys
from collections import Counter
from typing import List, Tuple
from aoc.helpers import locate, build_location, read_lines
from aoc.printer import get_meta_from_fn, print_


sys.setrecursionlimit(30_000)

@functools.cache
def quick_split_if_even(stone: int) -> List[int]:
    log_10, quotient = 1, stone
    while quotient := quotient // 10:
        log_10 += 1

    if log_10 % 2 == 1:
        return []

    quotient = 10 ** (log_10 // 2)
    top = stone // quotient
    top_floor = top * quotient
    bottom = stone - top_floor
    return [top, bottom]

class ObserverAutomaton:
    stones: Counter
    iteration: int
    end_on: int

    def __init__(self, stones: List[int], end_on: int):
        self.stones = Counter({s: 1 for s in stones})
        self.iteration = 0
        self.end_on = end_on

    def blink(self):
        materialized = dict(self.stones)
        self.stones.clear()
        for stone, amount in materialized.items():
            if stone == 0:
                self.stones.update({1: amount})
            elif len((maybe_two := quick_split_if_even(stone))) > 1:
                for new_stone in maybe_two:
                    self.stones.update({new_stone: amount})
            else:
                self.stones.update({stone * 2024: amount})

        self.iteration += 1

    def is_accepting(self):
        return self.end_on == self.iteration

    def sum_stones(self):
        return sum(v for _, v in self.stones.items())

def solve(_input=None):
    """
    :challenge: 55312
    :expect: 277444936413293
    """
    lines = []
    with open(locate(_input), "r") as fp:
        for line in read_lines(fp):
            stones = [int(d) for d in line.split(" ")]
            lines.append(stones)

    n: int = 25 if "test" in _input else 75
    observer = ObserverAutomaton(stones, n)  # 22
    while not observer.is_accepting():
        observer.blink()

    return observer.sum_stones()

```
## year_2024\day_11\solve_1.py

```py
import functools
import sys
from collections import Counter
from typing import List, Tuple
from aoc.helpers import locate, build_location, read_lines
from aoc.printer import get_meta_from_fn, print_


sys.setrecursionlimit(30_000)

@functools.cache
def quick_split_if_even(stone: int) -> List[int]:
    log_10, quotient = 1, stone
    while quotient := quotient // 10:
        log_10 += 1

    if log_10 % 2 == 1:
        return []

    quotient = 10 ** (log_10 // 2)
    top = stone // quotient
    top_floor = top * quotient
    bottom = stone - top_floor
    return [top, bottom]

class ObserverAutomaton:
    stones: Counter
    iteration: int
    end_on: int

    def __init__(self, stones: List[int], end_on: int):
        self.stones = Counter({s: 1 for s in stones})
        self.iteration = 0
        self.end_on = end_on

    def blink(self):
        materialized = dict(self.stones)
        self.stones.clear()
        for stone, amount in materialized.items():
            if stone == 0:
                self.stones.update({1: amount})
            elif len((maybe_two := quick_split_if_even(stone))) > 1:
                for new_stone in maybe_two:
                    self.stones.update({new_stone: amount})
            else:
                self.stones.update({stone * 2024: amount})

        self.iteration += 1

    def is_accepting(self):
        return self.end_on == self.iteration

    def sum_stones(self):
        return sum(v for _, v in self.stones.items())

def solve(__input=None):
    """
    :challenge: 55312
    :expect: 233875
    """
    lines = []
    with open(locate(__input), "r") as fp:
        for line in read_lines(fp):
            stones = [int(d) for d in line.split(" ")]
            lines.append(stones)

    observer = ObserverAutomaton(stones, 25)  # 22
    while not observer.is_accepting():
        observer.blink()

    return observer.sum_stones()

```
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
            lines.append(list(map(int, list(line))))

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

    def in_bounds(self, y: int, x: int) -> bool:
        return -1 < y < self.grid.shape[0] and -1 < x < self.grid.shape[1]

    def _step(self, y, x) -> Generator[Tuple[int, int, int], None, None]:
        steps = (
            (y - 1, x + 0),
            (y + 0, x + 1),
            (y + 1, x + 0),
            (y + 0, x + -1)
        )
        for step in steps:
            # In bounds of map and one elevation above
            if self.in_bounds(*step) and (value := self.grid[step]) == self.grid[y, x] + 1:
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
            lines.append(list(map(int, list(line))))

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
        for reshaped_blocks in reshape(blocks[1:]):
            yield reshaped_blocks

    elif free_head and len(blocks) > 1:
        number_tail, size_tail, free_tail = blocks[-1]

        if size_tail == free_head:
            yield number_tail, size_tail
            for reshaped_blocks in reshape(blocks[1:-1]):
                yield reshaped_blocks

        if free_head > size_tail:
            yield number_tail, size_tail
            new_blocks = [(0, 0, free_head - size_tail)] + blocks[1:-1]
            for reshaped_blocks in reshape(new_blocks):
                yield reshaped_blocks

        if size_tail > free_head:
            yield number_tail, free_head
            new_blocks = blocks[1:-1] + [(number_tail, size_tail - free_head, free_tail)]
            for reshaped_blocks in reshape(new_blocks):
                yield reshaped_blocks

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

def to_int(text: str):
    match text:
        case "#": return 1
        case "^": return 2
        case _: return 0

def solve_1(__input=None):
    """
    :challenge: 41
    :expect: 4973
    """
    _grid: List[List[int]] = []
    with open(locate(__input), "r") as fp:
        for ind, line in enumerate(read_lines(fp)):
            _grid.append([to_int(l) for l in list(line)])

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

    def process(self, commands: re.Match[str]):
        if commands[0] == "do()":
            self.multiply = True
        elif commands[0] == "don't()":
            self.multiply = False
        elif self.multiply:
            self.total += int(commands[1]) * int(commands[2])

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
