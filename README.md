# 🎄 Advent of Code 2022 🎄

- Install: `requirements install -r requirements.txt`
- Run `python3 year_2022/day_07/solve.py`
- Test: Use IDE :)
- Generate README: `python aoc/template.py`

*Note:* For browser automation: https://github.com/jmrnilsson/aoc-watcher


## year_2022/day_18/solve.py

```py
import json
import operator
import re
import sys
from collections import Counter, OrderedDict, defaultdict
import heapq as heap
import numpy as np
from aoc import tools
from aoc.helpers import timing, locate, Printer, build_location, test_nocolor, puzzle_nocolor, read_lines
from aoc.poll_printer import PollPrinter


def solve_1(input_=None):
    """
    test=58
    expect=1598415
    """
    is_test = 1 if "test" in input_ else 0

    pixels = set()
    neighbors = set()
    neighbors_n = 0

    with open(locate(input_), "r") as fp:
        for line in read_lines(fp):
            pixel = tuple(int(i) for i in re.findall(r"\d+", line))
            pixels.add(pixel)

    for p in pixels:
        for neighbor in von_neumann_neighbors_3d(p):
            if neighbor not in pixels:
                neighbors_n += 1

    return neighbors_n


def find_a_start_in_void(pixels, droplet_part):
    for pixel in pixels:
        for neighbor in von_neumann_neighbors_3d(pixel):
            try:
                if neighbor not in pixels:
                    x, y, z = neighbor
                    min_z = min(p[2] for p in pixels if x == p[0] and y == p[1])
                    max_z = max(p[2] for p in pixels if x == p[0] and y == p[1])
                    if min_z < neighbor[2] < max_z:
                        min_x = min(p[0] for p in pixels if y == p[1] and z == p[2])
                        max_x = max(p[0] for p in pixels if y == p[1] and z == p[2])
                        if min_x < neighbor[0] < max_x:
                            min_y = min(p[1] for p in pixels if z == p[2] and x == p[0])
                            max_y = max(p[1] for p in pixels if z == p[2] and x == p[0])
                            if min_y < neighbor[1] < max_y:
                                if neighbor not in droplet_part:
                                    return neighbor
            except ValueError:
                continue


def solve_2(input_=None):
    """
    test=58
    expect=1598415
    """
    is_test = 1 if "test" in input_ else 0

    pixels = set()
    droplet = set()
    surface_area = 0
    void_surface_area = 0
    void = 0

    with open(locate(input_), "r") as fp:
        for line in read_lines(fp):
            pixel = tuple(int(i) for i in re.findall(r"\d+", line))
            pixels.add(pixel)

    for p in pixels:
        for neighbor in von_neumann_neighbors_3d(p):
            if neighbor not in pixels:
                surface_area += 1

    # simplified dijkstra
    droplet = set()
    n = 0
    while start := find_a_start_in_void(pixels, droplet):

        residual = {start}
        seen = set()
        while residual:
            node = residual.pop()
            n += 1
            if n % 1000 == 0:
                print(f"queue size: {len(residual)}")
            seen.add(node)
            droplet.add(node)
            for neighbor in von_neumann_neighbors_3d(node):
                if neighbor not in seen and neighbor not in pixels:
                    residual.add(neighbor)

    for p in droplet:
        for neighbor in von_neumann_neighbors_3d(p):
            if neighbor not in droplet and neighbor:
                void_surface_area += 1

    # 3134 too high
    # 2226 too high
    return surface_area - void_surface_area


def von_neumann_neighbors_3d(p):
    return [
        (p[0] - 1, p[1], p[2]),
        (p[0] + 1, p[1], p[2]),
        (p[0], p[1] - 1, p[2]),
        (p[0], p[1] + 1, p[2]),
        (p[0], p[1], p[2] - 1),
        (p[0], p[1], p[2] + 1),
    ]

```
## year_2022/day_17/solve.py

```py
import json
import operator
import re
import sys
from collections import Counter, OrderedDict, defaultdict
from itertools import cycle
from typing import List, Tuple, Callable
import numpy as np
from bitarray import bitarray
from bitarray.util import ba2int
from aoc import tools
from aoc.helpers import timing, locate, Printer, build_location, test_nocolor, puzzle_nocolor, read_lines
from aoc.poll_printer import PollPrinter


def apply(coords: List[Tuple[int, int]], func: Callable):
    new_coords = []
    for coord in coords:
        new_coords.append(func(coord))
    return new_coords


def center(p, name):
    if name in ("horizontal_line", "reverse_l", "plus", "bloc"):
        return p[0] + 2, p[1]
    elif name in ("vertical_line"):
        return p[0] + 2, p[1]
    else:
        raise ValueError(f"Unknown name: {name}")


# simulate a rock falling down a cavern and bouncing off the walls and may be stopped by previous rocks that have fallen
# to the ground or on top of other rocks. Can't move right or left if blocked by a rock. Can only move down if not
# blocked by another rock. If blocked by a rock on the bottom it will stop, and another rock with different shape will
# fall.
def fall(movements, n=2023, is_test=True):
    horizontal_line = [(0, 0), (1, 0), (2, 0), (3, 0)]
    plus = [(0, 1), (1, 1), (2, 1), (1, 0), (1, 2)]
    reverse_l = [(0, 0), (1, 0), (2, 0), (2, 1), (2, 2)]
    vertical_line = [(0, 0), (0, 1), (0, 2), (0, 3)]
    bloc = [(0, 0), (1, 0), (0, 1), (1, 1)]

    # rocks = [np.asarray(a) for a in [horizontal_line, plus, reverse_l, vertical_line, bloc]]
    rocks_ = [horizontal_line, plus, reverse_l, vertical_line, bloc]
    rock_names = ["horizontal_line", "plus", "reverse_l", "vertical_line", "bloc"]
    rocks = [(n, rock)for n, rock in zip(rock_names, rocks_)]
    walls = {-1, 7}
    m = 0
    max_y = 0
    shape = set()
    down = lambda p: (p[0], p[1] - 1)

    wind = 0
    wind_mod = len(movements)
    for name, rock in cycle(rocks):
        rock = apply(rock, lambda p: center(p, name))
        out_of_bounds = 3 + max_y
        # + (1 if name in ["horizontal_line", "plus", "reverse_l"] else 0)
        up = lambda p: (p[0], p[1] + out_of_bounds)
        rock = apply(rock, up)
        y_min = min([p[1] for p in rock])
        assert y_min - max_y == 3
        last_rock = rock
        breaker = False
        while not breaker:
            movement = movements[wind % wind_mod]
            # for movement in movements:
            if movement == ">":
                sideways = lambda p: (p[0] + 1, p[1])
            else:
                sideways = lambda p: (p[0] - 1, p[1])
            before_wall = rock
            rock = apply(rock, sideways)
            for x, y in rock:
                if x in walls:
                    rock = before_wall
                elif (x, y) in shape:
                    rock = before_wall
            before_down = rock
            rock = apply(rock, down)
            for x, y in rock:
                if y == -1:
                    rock = before_down
                    breaker = True
                    # wind -= 1
                    # if wind < 0:
                    #     wind = len(movements) - 1
                    break
                elif (x, y) in shape:
                    rock = before_down
                    breaker = True
                    # wind -= 1
                    # if wind < 0:
                    #     wind = len(movements) - 1
                    break
            # if not breaker:
            wind += 1
            # else:
            #     wind -= 1
            #     if wind < 0:
            #         wind = len(movements) - 1

            last_rock = rock

        for coords in rock:
            if coords[1] in shape:
                raise ValueError(f"Duplicate y: {coords}")
            shape.add(coords)

        if is_test:
            if m == 0:
                assert rock == [(2, 0), (3, 0), (4, 0), (5, 0)]
            if m == 1:
                assert rock == [(2, 2), (3, 2), (4, 2), (3, 1), (3, 3)]
            if m == 2:
                assert rock == [(0, 3), (1, 3), (2, 3), (2, 4), (2, 5)]
            if m == 3:
                assert rock == [(4, 3), (4, 4), (4, 5), (4, 6)]
            if m == 4:
                assert rock == [(4, 7), (5, 7), (4, 8), (5, 8)]
            if m == 5:
                assert rock == [(1, 9), (2, 9), (3, 9), (4, 9)]

        max_y = max([y for _, y in shape]) + 1
        m += 1

        # _max_x = max([p[0] for p in shape])
        # display = np.zeros((max_y, _max_x + 2))
        #
        # for x, y in shape:
        #     display[y, x] = 1
        #
        # if 5 <= m < 100:
        #     d = np.rot90(display, 1)
        #     d = np.rot90(d, 1)
        #     d = np.fliplr(d)
        #     ys = [y for y in range(d.shape[0]) if np.any(d[y, :])]
        #     for y in ys:
        #         print("".join(["#" if d[y, x] else "." for x in range(d.shape[1])]))
        #
        #     print("\n------------------\n")

        if m == n - 1:
            break

    return abs(max_y)


def get_bytes(bits):
    done = False
    while not done:
        byte = 0
        for _ in range(0, 6):
            try:
                bit = next(bits)
            except StopIteration:
                bit = 0
                done = True
            byte = (byte << 1) | bit
        yield byte


def fall_p2(movements, n=2023, is_test=True):
    has_found_offset = False

    last_seen = dict()
    iters = defaultdict(set)

    horizontal_line = [(0, 0), (1, 0), (2, 0), (3, 0)]
    plus = [(0, 1), (1, 1), (2, 1), (1, 0), (1, 2)]
    reverse_l = [(0, 0), (1, 0), (2, 0), (2, 1), (2, 2)]
    vertical_line = [(0, 0), (0, 1), (0, 2), (0, 3)]
    bloc = [(0, 0), (1, 0), (0, 1), (1, 1)]

    # rocks = [np.asarray(a) for a in [horizontal_line, plus, reverse_l, vertical_line, bloc]]
    rocks_ = [horizontal_line, plus, reverse_l, vertical_line, bloc]
    rock_names = ["horizontal_line", "plus", "reverse_l", "vertical_line", "bloc"]
    rocks = [(n, rock)for n, rock in zip(rock_names, rocks_)]
    walls = {-1, 7}
    m = 0
    max_y = 0
    shape = set()
    down = lambda p: (p[0], p[1] - 1)

    wind = 0
    past_movements = []
    wind_mod = len(movements)
    rock_n = 0
    rock_mod = len(rocks)
    while 1:
        name, rock = rocks[rock_n % rock_mod]
        rock_n += 1
        # for name, rock in cycle(rocks):
        # for name, rock in cycle(rocks):
        rock = apply(rock, lambda p: center(p, name))
        out_of_bounds = 3 + max_y
        # + (1 if name in ["horizontal_line", "plus", "reverse_l"] else 0)
        up = lambda p: (p[0], p[1] + out_of_bounds)
        rock = apply(rock, up)
        y_min = min([p[1] for p in rock])
        assert y_min - max_y == 3
        breaker = False
        while not breaker:
            movement = movements[wind % wind_mod]
            past_movements += movement
            # for movement in movements:
            if movement == ">":
                sideways = lambda p: (p[0] + 1, p[1])
            else:
                sideways = lambda p: (p[0] - 1, p[1])
            before_wall = rock
            rock = apply(rock, sideways)
            for x, y in rock:
                if x in walls or (x, y) in shape:
                    rock = before_wall
                # elif (x, y) in shape:
                #    rock = before_wall
            before_down = rock
            rock = apply(rock, down)
            for x, y in rock:
                if y == -1:
                    rock = before_down
                    breaker = True
                    break
                elif (x, y) in shape:
                    rock = before_down
                    breaker = True
                    break
            wind += 1

        for coords in rock:
            if coords[1] in shape:
                raise ValueError(f"Duplicate y: {coords}")
            shape.add(coords)

        if is_test:
            if m == 0:
                assert rock == [(2, 0), (3, 0), (4, 0), (5, 0)]
            if m == 1:
                assert rock == [(2, 2), (3, 2), (4, 2), (3, 1), (3, 3)]
            if m == 2:
                assert rock == [(0, 3), (1, 3), (2, 3), (2, 4), (2, 5)]
            if m == 3:
                assert rock == [(4, 3), (4, 4), (4, 5), (4, 6)]
            if m == 4:
                assert rock == [(4, 7), (5, 7), (4, 8), (5, 8)]
            if m == 5:
                assert rock == [(1, 9), (2, 9), (3, 9), (4, 9)]

        max_y = max([y for _, y in shape]) + 1
        m += 1

        if not has_found_offset:
        # if m > 30 and not has_found_offset:
            _max_x = max([p[0] for p in shape])
            display = np.zeros((max_y, _max_x + 2), dtype=bool)

            for x, y in shape:
                display[y, x] = 1

            d = np.rot90(display, 1)
            d = np.rot90(d, 1)
            d = np.fliplr(d)
            # xs = d[:5000:,] 1715?
            xs = d[:500:,]  # 1715?
            asda = []
            for asdsad in xs:
                asda.append("".join([f"%d" % (1 if d else 0) for d in asdsad]))
            agg_x = [ba2int(bitarray(xx)) for xx in asda]
            # for xa in xsb:
            #     a = "!"
            #     b = a
            # ys = d[:,:3000]  # 1715?
            # sum_x = xs.sum(axis=0).tolist()
            # sum_y = xs.sum(axis=1).tolist()
            # canon_move = [d for d in past_movements[-500:]]

            key = "|".join([str(int(d)) for d in agg_x] + [name])  # + canon_move)
            iters[key].add(m)
            # key = tuple(sum_x + [name])
            if key in last_seen:
                cur = iters[key]

                offset = m - last_seen[key][0]
                y_diff = max_y - last_seen[key][1]
                print(f"Found a duplicate at {m}")
                print(f"Last seen at {last_seen[key][0]}")
                print(f"Diff offset: {offset}")
                print(f"Diff Y: {y_diff}")
                factor_mod = (1000000000000 - m) // offset
                # factor_mod -= 2
                y_diff_k = (y_diff * factor_mod)
                max_y += y_diff_k
                m += (factor_mod * offset)
                has_found_offset = True
                shape = set(apply(shape, lambda p: (p[0], p[1] + y_diff_k)))
            last_seen[key] = (m, max_y)


        # if 5 <= m < 100:
        #     d = np.rot90(display, 1)
        #     d = np.rot90(d, 1)
        #     d = np.fliplr(d)
        #     ys = [y for y in range(d.shape[0]) if np.any(d[y, :])]
        #     for y in ys:
        #         print("".join(["#" if d[y, x] else "." for x in range(d.shape[1])]))
        #
        #     print("\n------------------\n")

        if m == n - 1:
            break

    # test
    ## 1514285714288
    ## 1514285714288
    # 1580758017507
    # 1580758017507
    # 1580758017507
    # 1580758017507
    return abs(max_y)


def solve_1(input_=None):
    """
    test=58
    expect=1598415
    """
    is_test = 1 if "test" in input_ else 0

    seed = []
    seed_set = set()
    seed_dict = {}
    seed_counter = Counter()
    seed_ordered_dict = OrderedDict()
    seed_np = np.array([])
    seed_np_2d = np.array([[]])
    seed_default_dict = defaultdict(list)

    area = 0

    with open(locate(input_), "r") as fp:
        lines = list(read_lines(fp)[0])

    return fall(lines, 2023, bool(is_test))


def solve_2(input_=None):
    """
    test=34
    expect=3812909
    """
    is_test = 1 if "test" in input_ else 0

    with open(locate(input_), "r") as fp:
        lines = list(read_lines(fp)[0])

    return fall_p2(lines, 1000000000000, bool(is_test))


```
## year_2022/day_16/solve.py

```py
import copy
import operator
import heapq as heap
import operator
import re
import sys
from collections import defaultdict, Counter
from itertools import permutations
from queue import PriorityQueue
from typing import Set, Tuple, Dict, List
from more_itertools import sliding_window
from aoc import tools
from aoc.helpers import locate, build_location, read_lines
from aoc.poll_printer import PollPrinter


def dijkstra(graph, starting_node):
    visited = set()
    parents_map = {}
    pq = []
    node_costs = defaultdict(lambda: float('inf'))
    node_costs[starting_node] = 0
    heap.heappush(pq, (0, starting_node))

    while pq:
        # go greedily by always extending the shorter cost nodes first
        _, node = heap.heappop(pq)
        visited.add(node)

        for neighbor in graph[node]:
            weight = 1
            if neighbor in visited:
                continue

            new_cost = node_costs[node] + weight
            if node_costs[neighbor] > new_cost:
                parents_map[neighbor] = node
                node_costs[neighbor] = new_cost
                heap.heappush(pq, (new_cost, neighbor))

    return parents_map, node_costs


class RoundRobinSwapBeginFromHead(object):

    def __init__(self, length, from_n=0, to_n=0):
        self.length = length
        self.from_n = from_n
        self.to_n = to_n

    def _get_next_index(self):
        start_n = self.from_n
        while self.from_n == start_n or self.from_n == self.to_n:
            if self.from_n < self.length - 1:
                self.from_n += 1
            else:
                self.from_n = 0
                if self.to_n >= self.length - 1:
                    raise StopIteration
                self.to_n += 1

        return self.from_n, self.to_n

    def _sample_from(self, iterable: List):
        try:
            while 1:
                i, j = self._get_next_index()
                iterable = copy.deepcopy(iterable)
                swap = iterable[j]
                iterable[j] = iterable[i]
                iterable[i] = swap
                yield iterable
        except StopIteration:
            return

    def sample_from(self, *iterables: List[List]):
        for iterable in iterables:
            for sample in self._sample_from(iterable):
                yield sample


def solve_1(input_=None):
    """
    test=58
    expect=1598415
    """
    is_test = 1 if "test" in input_ else 0

    graph = {}
    flow_rates = {}
    nodes = []
    start = None

    with open(locate(input_), "r") as fp:
        for line in read_lines(fp):
            name = line[6:8]
            flow_rate = int(re.findall(r"rate=(\d+);", line)[0])
            leads_to = tuple(re.findall(r"[A-Z]{2}", line[10:]))
            graph[name] = leads_to
            flow_rates[name] = flow_rate
            nodes.append((flow_rate, name))
            if start is None:
                start = name

    sorted_nodes = sorted(nodes, key=lambda n: n, reverse=True)
    results = {}

    # sample = C
    _, costs = dijkstra(graph, nodes[0][1])
    reachable = set(costs.keys())

    meaningful_nodes: List[str] = [name for flow_rates, name in sorted_nodes if flow_rates and name in reachable]
    # state = CycleState(len(meaningful_nodes))

    n = 0
    # for node_set in [["DD", "BB", "JJ", "HH", "EE", "CC"]]:
    # for node_set in permutations(meaningful_nodes, len(meaningful_nodes)):
    round_robin_swap = RoundRobinSwapBeginFromHead(len(meaningful_nodes))
    priority_queue: List[Tuple[int, List[str]]] = [(0, meaningful_nodes)]
    seen = set()
    counter = Counter()

    while 1:
        str_last = ",".join(priority_queue[-1][1]) if priority_queue else None
        best = sorted([(v, k) for k, v in results.items()], key=lambda x: x[0], reverse=True)[0] if results else [0, []]
        str_best = ",".join(best[1])
        print(f"n: {n}, best_flow={best[0]}, best={str_best} last={str_last}")
        counter.update({best[0]: 1})
        if counter[best[0]] > 500 or not priority_queue:
            break

        pick = [priority_queue.pop(0)[1] for _ in range(min(3000, len(priority_queue)))]

        for node_set in round_robin_swap.sample_from(*pick):
            if tuple(node_set) in seen:
                continue
            seen.add(tuple(node_set))

            permutations_of_nodes = list(node_set)
            including_start = start, *[n for n in permutations_of_nodes if n != start]
            total_cost = 0
            total_flow_rate = 0
            accumulated_flow_rate = 0
            paths = []
            last = None
            for begin, end in sliding_window(including_start, 2):
                path, costs = dijkstra(graph, begin)
                cost = costs[end] + 1
                max_cost = min([30 - total_cost, cost])
                total_cost += cost
                begin_flow_rate = flow_rates[begin]
                accumulated_flow_rate += begin_flow_rate
                total_flow_rate += max_cost * accumulated_flow_rate
                paths.append(path)
                last = end
                if total_cost > 30:
                    break

            accumulated_flow_rate += flow_rates[last]
            residual = 30 - total_cost
            total_flow_rate += residual * accumulated_flow_rate
            results[tuple(permutations_of_nodes)] = total_flow_rate
            priority_queue.append((total_flow_rate, permutations_of_nodes))

            n += 1
            # if n % 10_000 == 0:
            #     print(f"n: {n}, best={max(results.values())}, last={permutations_of_nodes}")
            # print(f"total_flow_rate: {total_flow_rate}, path={permutations_of_nodes}")

    sorted_results = sorted(results.items(), key=lambda r: r[1], reverse=True)
    # a = sorted(results, key=lambda r: r[0], reverse=True)
    # print(sorted_results)
    # 1357 low
    # 1402 low
    return max(list(results.values()) + [0])


def solve_2(input_=None):
    """
    test=34
    expect=3812909
    """
    start = 1 if "test" in input_ else 0

    total = 0
    area = 0

    with open(locate(input_), "r") as fp:
        lines = read_lines(fp)[:start]

    matrix = tools.to_matrix(lines, " ", replace_chars="[]")
    tools.bitmap_print_matrix(matrix)
    print(tools.get_vectors(matrix, str_join=True))

    for line in lines:
        if line == "\n":
            continue
        l, w, h = [int(i) for i in re.findall(r"\d+", line)]
        area += operator.add(*sorted([l, w, h])[:-1])*2
        area += l * w * h

    return total


```
## year_2022/day_15/solve.py

```py
import operator
import re
import sys
from typing import List, Tuple, Set, Callable
from aoc.helpers import locate, build_location, read_lines
from aoc.poll_printer import PollPrinter


def manhattan(sensor, beacon):
    return abs(sensor[0] - beacon[0]) + abs(sensor[1] - beacon[1])


def manhattan_traverse_y(start_x: int, end_x: int, items: List[Tuple[int, int]], beacons_in_x: Set[int], sensors_in_x: Set[int], is_test: bool):
    xs = set()
    for x in range(start_x, end_x + 1):
        for sensor_x, distance_in_x in items:
            if sensor_x - distance_in_x <= x <= sensor_x + distance_in_x:
                if x not in beacons_in_x and x not in sensors_in_x:
                    xs.add(x)
                    break

    return sum(1 for _ in xs)


def solve_1(input_=None):
    """
    test=26
    expect=5256611
    """
    is_test = 1 if "test" in input_ else 0
    seed = []

    with open(locate(input_), "r") as fp:
        for line in read_lines(fp):
            digits = [int(i) for i in re.findall(r"-?\d+", line)]
            sensor = tuple(digits[:2])
            beacon = tuple(digits[2:])
            seed.append((sensor, beacon))

    y = 10 if is_test else 2000000
    parameters = []
    for sensor, beacon in seed:
        distance = manhattan(sensor, beacon)
        distance_in_x = distance - abs(sensor[1] - y)
        parameter = sensor[0], sensor[1], beacon[0], beacon[1], distance_in_x
        parameters.append(parameter)

    min_x = min([point[0] - point[4] for point in parameters]) - 1
    max_x = max([point[0] + point[4] for point in parameters]) + 1
    items = [(p[0], p[4]) for p in parameters]
    beacons_in_x = {p[2] for p in parameters if p[3] == y}
    sensors_in_x = {p[0] for p in parameters if p[1] == y}

    return manhattan_traverse_y(min_x, max_x, items, beacons_in_x, sensors_in_x, bool(is_test))


def find_beacon(x, y, seed):
    found = True

    for sensor, beacon in seed:
        distance = abs(sensor[0] - beacon[0]) + abs(sensor[1] - beacon[1])
        distance_in_x = distance - abs(sensor[1] - y)
        if distance_in_x > 0:
            if sensor[0] - distance_in_x <= x <= sensor[0] + distance_in_x:
                found = False
                break

    if found:
        return x * 4000000 + y

    return None


def next_edge(operator_x: Callable, operator_y: Callable, sensor: Tuple[int, int], k, min_k, max_k):
    edge = operator_x(sensor[0], k), operator_y(sensor[1],  k)
    if min_k <= edge[0] <= max_k:
        if min_k <= edge[1] <= max_k:
            return edge
    return None


def every_von_neumann_edge(max_k, min_k, seed: List[Tuple[Tuple[int, int], Tuple[int, int]]]):
    for sensor, beacon in seed:
        distance = abs(sensor[0] - beacon[0]) + abs(sensor[1] - beacon[1]) + 1
        for k in range(0, distance + 1):
            for op_x, op_y in [(operator.sub, operator.add), (operator.add, operator.add)]:
                sensor_up = sensor[0], sensor[1] - distance
                if edge := next_edge(op_x, op_y, sensor_up, k, min_k, max_k):
                    yield edge
            for op_x, op_y in [(operator.sub, operator.sub), (operator.add, operator.sub)]:
                sensor_down = sensor[0], sensor[1] + distance
                if edge := next_edge(op_x, op_y, sensor_down, k, min_k, max_k):
                    yield edge


def solve_2(input_=None):
    """
    test=56000011
    expect=13337919186981
    """
    is_test = 1 if "test" in input_ else 0
    seed: List[Tuple, Tuple] = []

    with open(locate(input_), "r") as fp:
        for line in read_lines(fp):
            digits = [int(i) for i in re.findall(r"-?\d+", line)]
            sensor = tuple(digits[:2])
            beacon = tuple(digits[2:])
            seed.append((sensor, beacon))

    min_k = 0
    max_k = 4000000 if not is_test else 20

    for x, y in every_von_neumann_edge(max_k, min_k, seed):
        if find_beacon(x, y, seed):
            if 0 <= x <= max_k:
                if 0 <= y <= max_k:
                    print(f"y: {y} x: {x}")
                    return x * 4000000 + y

    return None



```
## year_2022/day_14/solve.py

```py
import abc
import sys
from typing import Tuple, List, Callable
import numpy as np
from aoc.helpers import locate, build_location, read_lines
from aoc.poll_printer import PollPrinter


class SandPhysics(object):

    def __init__(self, matrix):
        self.matrix = matrix

    @abc.abstractmethod
    def down(self, sand):
        pass

    @abc.abstractmethod
    def left_or_right(self, sand):
        pass

    def pour(self, sand):
        previous = None
        while previous != sand:
            previous = sand
            sand = self.down(sand)
            sand = self.left_or_right(sand)
        return sand


class SandPhysicsPart1(SandPhysics):
    def left_or_right(self, sand):
        x, y = sand
        if self.matrix[y + 1, x - 1] == 0:
            y += 1
            x -= 1
        elif self.matrix[y + 1, x + 1] == 0:
            y += 1
            x += 1
        return x, y

    def down(self, sand):
        x, y = sand
        has_changed = False
        while self.matrix[y, x] == 0:
            y += 1
            has_changed = True
        return x, (y - 1 if has_changed else y)


class SandPhysicsPart2(SandPhysics):

    def left_or_right(self, sand):
        x, y = sand
        if len(self.matrix) > y + 1:
            if self.matrix[y + 1, x - 1] == 0:
                y += 1
                x -= 1
            elif self.matrix[y + 1, x + 1] == 0:
                y += 1
                x += 1
            return x, y
        return sand

    def down(self, sand):
        x, y = sand
        pristine = True
        while len(self.matrix) > y + 1 and self.matrix[y, x] == 0:
            y += 1
            pristine = False
        return x, (y - 1 if not pristine else y)


def parse_line(line):
    previous = None
    for split_line in line.split(" -> "):
        x, y = [int(d) for d in split_line.split(",")]
        if previous:
            if previous[0] == x:
                first, second = sorted([previous[1], y])
                for y_ in range(first, second + 1):
                    yield x, y_
            if previous[1] == y:
                first, second = sorted([previous[0], x])
                for x_ in range(first, second + 1):
                    yield x_, y
        previous = x, y


def pour_sand(seed: List[Tuple[int, int]], sand_physics_factory: Callable, expand_x: int = 0, expand_y: int = 0):
    max_x, max_y = max(x for x, _ in seed),  max(y for _, y in seed)
    matrix = np.zeros((max_y + 1 + expand_y, max_x + 1 + expand_x))

    for x, y in seed:
        matrix[y, x] = 1

    n, sand_physics, entry_point = 0, sand_physics_factory(matrix), (500, 0)
    while 1:
        try:
            n += 1
            x, y = sand_physics.pour(entry_point)
            matrix[y, x] = 2
        except IndexError:
            break
        if (x, y) == entry_point:
            break
    return n


def solve_1(input_=None):
    """
    test=24
    expect=897
    """
    seed = []

    with open(locate(input_), "r") as fp:
        for line in read_lines(fp):
            seed += list(parse_line(line))

    return pour_sand(seed, lambda mat: SandPhysicsPart1(mat)) - 1  # The overflow sand not counted in the first part


def solve_2(input_=None):
    """
    test=93
    expect=26683
    """
    seed = []

    with open(locate(input_), "r") as fp:
        for line in read_lines(fp):
            seed += list(parse_line(line))

    return pour_sand(seed, lambda mat: SandPhysicsPart2(mat), expand_y=1, expand_x=170)


```
## year_2022/day_13/solve.py

```py
import json
import operator
import sys
from enum import Enum
from functools import cmp_to_key
from itertools import zip_longest
from more_itertools import chunked
from aoc.helpers import locate, build_location, read_lines
from aoc.poll_printer import PollPrinter


class Comparison(Enum):
    LT = -1
    EQ = 0
    GT = 1


def ok_pair(left, right):
    if left > right:
        return Comparison.GT
    if left < right:
        return Comparison.LT

    return Comparison.EQ


def ok_all(left_, right_):
    for left, right in zip_longest(left_, right_, fillvalue=-1):
        if isinstance(left, list) and isinstance(right, list):
            if (comparison := ok_all(left, right)) != Comparison.EQ:
                return comparison

        elif isinstance(left, int) and isinstance(right, list):
            if left == -1:
                return Comparison.LT
            if (comparison := ok_all([left], right)) != Comparison.EQ:
                return comparison

        elif isinstance(left, list) and isinstance(right, int):
            if right == -1:
                return Comparison.GT
            if (comparison := ok_all(left, [right])) != Comparison.EQ:
                return comparison

        elif isinstance(left, int) and isinstance(right, int):
            if (comparison := ok_pair(left, right)) != Comparison.EQ:
                return comparison

    return Comparison.EQ


def solve_1(input_=None):
    """
    test=13
    expect=6478
    """
    is_test = 1 if "test" in input_ else 0

    seed = []

    with open(locate(input_), "r") as fp:
        for signal in chunked(read_lines(fp), 2):
            left = json.loads(signal[0])
            right = json.loads(signal[1])
            seed.append((left, right))

    n, correct = 0, []
    for left, right in seed:
        n += 1
        if isinstance(left, int) and isinstance(right, int):
            if ok_pair(left, right) == Comparison.LT:
                correct.append(n)
        elif isinstance(left, list) and isinstance(right, list):
            if ok_all(left, right) == Comparison.LT:
                correct.append(n)
    return sum(correct)


def compare(left, right):
    if comparison := ok_all(left, right) == Comparison.LT:
        return -1
    if comparison == Comparison.GT:
        return 1

    return 0


def solve_2(input_=None):

    """
    test=140
    expect=21922
    """

    seed = []

    with open(locate(input_), "r") as fp:
        for signal in read_lines(fp):
            s = json.loads(signal)
            seed.append(s)

    splicers = [[2]], [[6]]
    seed += list(splicers)
    seed.sort(key=cmp_to_key(compare))
    indices = seed.index(splicers[0]) + 1, seed.index(splicers[1]) + 1

    return operator.mul(*indices)


```
## year_2022/day_12/solve.py

```py
import sys
from collections import defaultdict
import numpy as np
from aoc.helpers import locate, build_location, read_lines
from aoc.poll_printer import PollPrinter


class Node:
    def __init__(self, position: (), parent: ()):
        self.position = position
        self.parent = parent
        self.g = 0  # Distance to start node
        self.h = 0  # Distance to goal node
        self.f = 0  # Total cost

    def __eq__(self, other):
        return self.position == other.position

    def __lt__(self, other):
        return self.f < other.f

    def __repr__(self):
        return '({0},{1})'.format(self.position, self.f)


def astar_search(_maze: np.ndarray, start, end, weight_min=0):
    def is_closer(_open, _neighbor):
        for _node in _open:
            if _neighbor == _node and _neighbor.f >= _node.f:
                return False
        return True

    open_ = []
    closed = []
    start_node = Node(start, None)
    goal_node = Node(end, None)
    node_costs = defaultdict(lambda: float('inf'))
    open_.append(start_node)
    y_len, x_len = len(_maze), len(_maze[0])

    while len(open_) > 0:
        open_.sort()
        current_node = open_.pop(0)
        closed.append(current_node)

        current_node_position = current_node.position
        goal_node_position = goal_node.position
        if current_node_position == goal_node_position:
            path = []
            while current_node != start_node:
                path.append(current_node.position)
                current_node = current_node.parent

            return path[::-1]

        x, y = current_node.position
        neighbors = [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]
        current_weight = _maze[x][y]

        for n in neighbors:
            if 0 > n[0] or y_len <= n[0] or 0 > n[1] or x_len <= n[1]:
                continue

            if not (weight_ := _maze[n[0]][n[1]]):
                continue

            if not (weight_min <= weight_ < current_weight + 2):
                continue

            neighbor = Node(n, current_node)

            if neighbor in closed:
                continue

            cost = 1 + (1 if current_weight != weight_ else 0)

            total_cost = current_node.g + cost
            node_costs[neighbor.position] = min(node_costs[neighbor.position], total_cost)

            # G = combined_cost or combined_cost / count
            neighbor.g = node_costs[neighbor.position]
            neighbor.h = abs(neighbor.position[0] - goal_node.position[0])
            neighbor.h += abs(neighbor.position[1] - goal_node.position[1])
            neighbor.f = neighbor.g + neighbor.h

            if is_closer(open_, neighbor):
                open_.append(neighbor)

    return None


def solve_1(input_=None):
    """
    test=31
    expect=490
    """
    start = None
    end = None

    with open(locate(input_), "r") as fp:
        lines = read_lines(fp)

    maze = np.ones((len(lines), len(lines[0])), dtype=np.int8)

    for y, line in enumerate(lines):
        for x, ch in enumerate(list(line)):
            if ch == "S":
                start = (y, x)
            elif ch == "E":
                end = (y, x)
            else:
                maze[y][x] = ord(ch) - 96

    maze[end[0]][end[1]] = ord("z") - 96 + 1

    path = astar_search(maze, start, end)
    return sum(1 for _ in path)


def solve_2(input_=None):
    """
    test=29
    expect=488
    """
    start, end = None, None

    with open(locate(input_), "r") as fp:
        lines = read_lines(fp)

    maze = np.ones((len(lines), len(lines[0])), dtype=np.int8)

    for y, line in enumerate(lines):
        for x, ch in enumerate(list(line)):
            if ch == "E":
                end = (y, x)
            else:
                maze[y][x] = ord(ch) - 96

    maze[end[0]][end[1]] = ord("z") - 96 + 1
    starting_positions_func = np.vectorize(lambda t: t == 1)
    starting_positions = {(x, y) for x, y in np.argwhere(starting_positions_func(maze))}

    shortest = float('inf')
    for start in starting_positions:
        if not (path := astar_search(maze, start, end, weight_min=2)):
            continue
        current = sum(1 for _ in path)
        shortest = min(shortest, current)
        print(f"shortest: {shortest}, current: {current}")

    return shortest


```
## year_2022/day_11/solve.py

```py
import math
import math
import re
import sys
from enum import Enum
from functools import reduce
from typing import Tuple, List
from more_itertools import chunked
from aoc.helpers import locate, build_location, read_lines
from aoc.poll_printer import PollPrinter


class Monkey(object):
    def __init__(self, name: int, items: List[int], operation, divisor: int, targets: Tuple[int, int]):
        self.name: int = name
        self.operation = operation
        self.items: List[int] = items
        self.inspections = 0
        self.test_divisor: int = divisor
        self.targets: Tuple[int, int] = targets
        self.other_monkeys = None
        self.lcm = None
        self.custom_divisor = None

    def __repr__(self):
        return self.name

    def throw_at(self, item):
        self.items.append(item)

    def befriend(self, other_monkeys):
        self.other_monkeys = other_monkeys

    def add_lcm(self, _lcm):
        self.lcm = _lcm

    def business(self):
        for item in self.items:
            self.inspections += 1
            worry_level = self.operation(item)
            worry_modulus = worry_level % self.lcm

            if worry_level > self.lcm * 2:
                worry_level = self.lcm + worry_modulus

            if self.custom_divisor:
                worry_level //= self.custom_divisor

            if worry_level // self.test_divisor == worry_level / self.test_divisor:
                self.other_monkeys[self.targets[0]].throw_at(worry_level)
            else:
                self.other_monkeys[self.targets[1]].throw_at(worry_level)

        self.items.clear()


def lcm(a, b):
    return abs(a * b) // math.gcd(a, b)


class Day11Operation(Enum):
    POW2 = r"new = old \* old"
    MUL = r"new = old \* (\d+)"
    ADD = r"new = old \+ (\d+)"


def parse_monkey(monkey: List[str]):
    name = int(re.findall(r"(\d+)", monkey[0])[0])
    items = [int(i) for i in re.findall(r"(\d+)", monkey[1])]
    operation = None
    for operation_match in list(Day11Operation):
        if match := re.findall(operation_match.value, monkey[2]):
            if operation_match == Day11Operation.POW2:
                operation = lambda old: math.pow(old, 2)
            elif operation_match == Day11Operation.MUL:
                operation = lambda old: old * int(match[0])
            elif operation_match == Day11Operation.ADD:
                operation = lambda old: old + int(match[0])
            break
    assert operation
    divisor = int(re.findall(r"(\d+)", monkey[3])[0])
    targets = int(re.findall(r"(\d+)", monkey[4])[0]), int(re.findall(r"(\d+)", monkey[5])[0])
    return Monkey(name, items, operation, divisor, targets)


def do_monkey_business(monkeys: List[Monkey], n: int):
    divisors = []
    for monkey in monkeys:
        divisors.append(monkey.test_divisor)
        monkey.befriend(monkeys)

    _lcm = reduce(lcm, divisors)

    for monkey in monkeys:
        monkey.add_lcm(_lcm)

    for n in range(0, n):
        for monkey in monkeys:
            monkey.business()

    inspections = [m.inspections for m in monkeys]
    inspections.sort(key=lambda x: x, reverse=True)
    return inspections[0] * inspections[1]


def solve_1(input_=None):
    """
    test=10605
    expect=54253
    """
    monkeys: List[Monkey] = []

    with open(locate(input_), "r") as fp:
        for monkey in chunked(read_lines(fp), 6):
            monkey_object = parse_monkey(monkey)
            monkeys.append(monkey_object)
            monkey_object.custom_divisor = 3

    return do_monkey_business(monkeys, 20)


def solve_2(input_=None):
    """
    test=2713310158
    expect=13119526120
    """
    monkeys: List[Monkey] = []

    with open(locate(input_), "r") as fp:
        for monkey in chunked(read_lines(fp), 6):
            monkey_object = parse_monkey(monkey)
            monkeys.append(monkey_object)

    return do_monkey_business(monkeys, 10_000)


```
## year_2022/day_10/solve.py

```py
import re
import re
import sys
from enum import Enum
from typing import List, Tuple, Dict
from defaultlist import defaultlist
from aoc.helpers import locate, build_location, read_lines
from aoc.poll_printer import PollPrinter


class Day10Operation(Enum):
    ADDX = "addx"
    NOOP = "noop"


def compute(instructions: List[Tuple[Day10Operation, int]], until=260):
    x_s: List[int] = []
    x: int = 1
    i: int = 0
    op: Day10Operation
    number: int
    thread_join: Dict[int, int] = {}
    for n in range(0, until):
        x_s.append(x)
        canonical_x = thread_join.get(n)
        if canonical_x:
            x = canonical_x
            continue

        op, digit = instructions[i] if i < len(instructions) else (Day10Operation.NOOP, -1)
        if op == Day10Operation.ADDX:
            thread_join[n + 1] = x + digit
            i += 1
        elif op == Day10Operation.NOOP:
            i += 1
    return x_s


def solve_1(input_=None):
    """
    test=13140
    expect=13520
    """
    is_test = 1 if "test" in input_ else 0

    lines: List[Tuple[Day10Operation, int]] = []
    with open(locate(input_), "r") as fp:
        for line in read_lines(fp):
            operation = re.findall(r"^(\w+)?", line)[0]
            number = int(line.replace(operation, "").strip()) if operation == "addx" else -1
            lines.append((Day10Operation(operation), number))

    x_s: List[int] = compute(lines, until=230)
    signal = [v * (n + 1) for n, v in enumerate(x_s)]
    at = [20, 60, 100, 140, 180, 220]
    selected = [(i + 1, v) for i, v in enumerate(signal) if i + 1 in at]

    return sum(v for _, v in selected)


def solve_2(input_=None):
    """
    test=##..##..##..##..##..##..##..##..##..##..
###...###...###...###...###...###...###.
####....####....####....####....####....
#####.....#####.....#####.....#####.....
######......######......######......####
#######.......#######.......#######.....
    expect=###...##..###..#..#.###..####..##..###..
#..#.#..#.#..#.#..#.#..#.#....#..#.#..#.
#..#.#....#..#.####.###..###..#..#.###..
###..#.##.###..#..#.#..#.#....####.#..#.
#....#..#.#....#..#.#..#.#....#..#.#..#.
#.....###.#....#..#.###..####.#..#.###..
    """
    is_test = 1 if "test" in input_ else 0

    lines: List[Tuple[Day10Operation, int]] = []
    with open(locate(input_), "r") as fp:
        for line in read_lines(fp):
            operation = re.findall(r"^(\w+)?", line)[0]
            number = int(line.replace(operation, "").strip()) if operation == "addx" else -1
            lines.append((Day10Operation(operation), number))

    x_s = compute(lines, until=260)

    acc = defaultlist(list)
    default_sprite = "........................................"
    for i in range(0, len(x_s)):
        x = x_s[i]
        write_out = [x - 1, x, x + 1]
        mod_sprite = list(default_sprite)
        for w in write_out:
            mod_sprite[w] = "#"
        sprite = "".join(mod_sprite)
        pixel_index, row = i % len(sprite), i // len(sprite)
        acc[row].append(sprite[pixel_index])

    return "\n" + "\n".join(["".join(row) for row in acc[:6]])


```
## year_2022/day_09/solve.py

```py
import sys
from typing import Tuple, Set, List, Dict
import numpy as np
from aoc.helpers import locate, build_location, read_lines
from aoc.poll_printer import PollPrinter


def move(direction, position, amount, limit=None):
    limited_amount = min(amount, limit) if limit else amount
    if direction == "N":
        return position[0], position[1] + limited_amount
    if direction == "S":
        return position[0], position[1] - limited_amount
    if direction == "E":
        return position[0] + limited_amount, position[1]
    if direction == "W":
        return position[0] - limited_amount, position[1]
    raise ValueError(f"Cannot go to there {direction}")


def follow(previous, current):
    y, x = previous
    tail_y, tail_x = current
    if tail_y not in [y, y + 1, y - 1] or tail_x not in [x, x + 1, x - 1]:
        if tail_y < y:
            tail_y += 1
        elif tail_y > y:
            tail_y -= 1
        if tail_x < x:
            tail_x += 1
        elif tail_x > x:
            tail_x -= 1

    return tail_y, tail_x


def solve_1(input_=None):
    """
    test=13
    expect=6314
    """
    test = 1 if "test" in input_ else 0
    lines: List[Tuple[str, str, int]] = []
    position: Tuple[int, int] = 0, 0
    tail_position: Tuple[int, int] = position
    trail: Set[Tuple[int, int]] = {tail_position}
    translation: Dict[str, str] = {"U": "N", "D": "S", "L": "W", "R": "E"}

    with open(locate(input_), "r") as fp:
        for line in read_lines(fp):
            silly_direction, amount = line.split(" ")
            proper_direction = translation[silly_direction]
            lines.append((proper_direction, silly_direction, int(amount)))

    n = 0
    for direction, silly_direction, amount in lines:
        destination = move(direction, position, amount)

        while position != destination:
            position = move(direction, position, amount, 1)

            tail_position = follow(position, tail_position)
            trail.add(tail_position)

            if test:
                matrix_out = np.zeros((6, 5))
                matrix_out[position] = 1
                matrix_out[tail_position] = 2
                tr = np.transpose(matrix_out)
                tr = np.flip(tr, 0)
                print(f"{n}: {silly_direction} {amount}\n{tr}\n")
                n += 1

    return sum(1 for _ in trail)


def solve_2(input_=None):
    """
    test=36
    expect=2504
    """
    lines = []
    position: Tuple[int, int] = 0, 0
    tail_positions: List[Tuple[int, int]] = [position for _ in range(0, 9)]
    assert len(tail_positions) == 9
    trail: Set[Tuple[int, int]] = set(tail_positions)
    translation: Dict[str, str] = {"U": "N", "D": "S", "L": "W", "R": "E"}

    with open(locate(input_), "r") as fp:
        for line in read_lines(fp):
            silly_direction, amount = line.split(" ")
            proper_direction = translation[silly_direction]
            lines.append((proper_direction, silly_direction, int(amount)))

    for direction, _, amount in lines:
        destination = move(direction, position, amount)

        while position != destination:
            position = move(direction, position, amount, 1)

            previous = position
            materialized = list(tail_positions)
            tail_positions.clear()

            for t in range(0, 9):
                previous = follow(previous, materialized[t])
                tail_positions.append(previous)

                if t == 8:
                    trail.add(previous)

    return sum(1 for _ in trail)


```
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
    test=21
    expect=1713
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
    test=8
    expect=268464
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
    test=24933642
    expect=6296435
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
_default_puzzle_input = 'year_2022/day_01/puzzle.txt'
_default_test_input = 'year_2022/day_01/test.txt'

puzzle_input = build_location(__file__, 'puzzle.txt')
test_input = build_location(__file__, 'test.txt')
test_input_2 = build_location(__file__, 'test_2.txt')
*/

const test_input = 'aoc\\year_2022\\day_03_extras_javascript\\test.txt';
const puzzle_input = 'aoc\\year_2022\\day_03_extras_javascript\\puzzle.txt';

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
    const name = args[0].match('(test|puzzle)', 'i')[1];
    return `${func_name} - ${name.padEnd(6,' ')} (${ms}ms):`.padEnd(33,' ') + answer;
}


async function start(argv) {
    console.log(await timed('solve1', test_input));
    console.log(await timed('solve1', puzzle_input));
    console.log(await timed('solve2', test_input));
    console.log(await timed('solve2', puzzle_input));
    console.log('\nDone!');
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

