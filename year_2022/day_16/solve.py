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

# ICE
_default_puzzle_input = "year_2022/day_01/puzzle.txt"
_default_test_input = "year_2022/day_01/test.txt"

puzzle_input = build_location(__file__, "puzzle.txt")
test_input = build_location(__file__, "test.txt")
test_input_2 = build_location(__file__, "test_2.txt")

challenge_solve_1 = "exception"
challenge_solve_2 = "exception"


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


if __name__ == "__main__":
    """
    PWSH
    while ($true) {python -m aoc.year_2022.day_03.solve -poll; start-sleep -seconds 1} 

    BASH
    watch -n 3 python3 -m aoc.year_2021.day_03.day -p
    """
    poll_printer = PollPrinter(solve_1, solve_2, challenge_solve_1, challenge_solve_2, test_input, puzzle_input)
    poll_printer.run(sys.argv)

