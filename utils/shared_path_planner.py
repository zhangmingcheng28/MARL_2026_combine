import heapq
import itertools
from collections import defaultdict


def _to_cell(cell):
    return int(cell[0]), int(cell[1])


def _heuristic(cell, goal):
    return abs(cell[0] - goal[0]) + abs(cell[1] - goal[1])


def _grid_shape(binary_map):
    return len(binary_map), len(binary_map[0])


def _is_blocked(binary_map, cell):
    return binary_map[cell[0]][cell[1]] != 0


def _build_constraint_tables(constraints, agent_id):
    vertex_constraints = defaultdict(set)
    edge_constraints = defaultdict(set)
    max_constraint_time = 0

    for constraint in constraints:
        if constraint["agent"] != agent_id:
            continue

        max_constraint_time = max(max_constraint_time, constraint["time"])
        if constraint["kind"] == "vertex":
            vertex_constraints[constraint["time"]].add(_to_cell(constraint["cell"]))
        elif constraint["kind"] == "edge":
            edge_constraints[constraint["time"]].add(
                (_to_cell(constraint["from"]), _to_cell(constraint["to"]))
            )

    return vertex_constraints, edge_constraints, max_constraint_time


def _reconstruct_path(came_from, state):
    path = [state[0]]
    while state in came_from:
        state = came_from[state]
        path.append(state[0])
    path.reverse()
    return path


def _constrained_astar(binary_map, start, goal, constraints, agent_id, allow_wait):
    start = _to_cell(start)
    goal = _to_cell(goal)
    rows, cols = _grid_shape(binary_map)
    vertex_constraints, edge_constraints, max_constraint_time = _build_constraint_tables(constraints, agent_id)

    if not (0 <= start[0] < rows and 0 <= start[1] < cols):
        return []
    if not (0 <= goal[0] < rows and 0 <= goal[1] < cols):
        return []
    if _is_blocked(binary_map, start) or _is_blocked(binary_map, goal):
        return []
    if start in vertex_constraints.get(0, set()):
        return []

    moves = ((0, 1), (1, 0), (0, -1), (-1, 0))
    if allow_wait:
        moves = moves + ((0, 0),)

    slack = max(rows * cols, 32)
    max_time = rows * cols + max_constraint_time + slack
    open_set = []
    came_from = {}
    g_score = {(start, 0): 0}
    best_seen = {}
    counter = itertools.count()
    heapq.heappush(open_set, (_heuristic(start, goal), next(counter), 0, start))

    while open_set:
        _, _, current_time, current = heapq.heappop(open_set)
        state = (current, current_time)
        current_cost = g_score[state]

        if best_seen.get(state, float("inf")) <= current_cost:
            continue
        best_seen[state] = current_cost

        if current == goal and current_time >= max_constraint_time:
            return _reconstruct_path(came_from, state)

        if current_time >= max_time:
            continue

        for row_delta, col_delta in moves:
            next_cell = (current[0] + row_delta, current[1] + col_delta)
            next_time = current_time + 1

            if not (0 <= next_cell[0] < rows and 0 <= next_cell[1] < cols):
                continue
            if _is_blocked(binary_map, next_cell):
                continue
            if next_cell in vertex_constraints.get(next_time, set()):
                continue
            if (current, next_cell) in edge_constraints.get(current_time, set()):
                continue

            next_state = (next_cell, next_time)
            tentative_cost = current_cost + 1
            if tentative_cost >= g_score.get(next_state, float("inf")):
                continue

            came_from[next_state] = state
            g_score[next_state] = tentative_cost
            f_score = tentative_cost + _heuristic(next_cell, goal)
            heapq.heappush(open_set, (f_score, next(counter), next_time, next_cell))

    return []


def astar_path(binary_map, start, goal):
    return _constrained_astar(binary_map, start, goal, [], 0, False)


def compress_path_turns(path):
    if not path:
        return []

    normalized_path = [_to_cell(cell) for cell in path]
    if len(normalized_path) <= 2:
        return normalized_path

    refined_path = [normalized_path[0]]
    current_heading = (
        normalized_path[1][0] - normalized_path[0][0],
        normalized_path[1][1] - normalized_path[0][1],
    )

    for path_idx in range(2, len(normalized_path)):
        next_heading = (
            normalized_path[path_idx][0] - normalized_path[path_idx - 1][0],
            normalized_path[path_idx][1] - normalized_path[path_idx - 1][1],
        )
        if next_heading != current_heading:
            refined_path.append(normalized_path[path_idx - 1])
            current_heading = next_heading

    refined_path.append(normalized_path[-1])
    return refined_path


def _path_position(path, time_step):
    if time_step < len(path):
        return path[time_step]
    return path[-1]


def _detect_conflict(paths):
    if not paths:
        return None

    max_path_len = max(len(path) for path in paths)
    for time_step in range(max_path_len):
        occupied = {}
        for agent_id, path in enumerate(paths):
            cell = _path_position(path, time_step)
            if cell in occupied:
                return {
                    "kind": "vertex",
                    "time": time_step,
                    "a1": occupied[cell],
                    "a2": agent_id,
                    "cell": cell,
                }
            occupied[cell] = agent_id

        if time_step == max_path_len - 1:
            continue

        traversed_edges = {}
        for agent_id, path in enumerate(paths):
            edge = (_path_position(path, time_step), _path_position(path, time_step + 1))
            reverse_edge = (edge[1], edge[0])
            if reverse_edge in traversed_edges:
                return {
                    "kind": "edge",
                    "time": time_step,
                    "a1": traversed_edges[reverse_edge],
                    "a2": agent_id,
                    "from": edge[0],
                    "to": edge[1],
                }
            traversed_edges[edge] = agent_id

    return None


def cbs_plan_paths(binary_map, starts, goals):
    starts = [_to_cell(start) for start in starts]
    goals = [_to_cell(goal) for goal in goals]
    if len(starts) != len(goals):
        raise ValueError("CBS requires the same number of starts and goals.")

    root_paths = []
    root_constraints = []
    for agent_id, (start, goal) in enumerate(zip(starts, goals)):
        path = _constrained_astar(binary_map, start, goal, root_constraints, agent_id, True)
        if not path:
            return []
        root_paths.append(path)

    open_set = []
    counter = itertools.count()
    root_cost = sum(len(path) - 1 for path in root_paths)
    heapq.heappush(
        open_set,
        (root_cost, next(counter), {"constraints": root_constraints, "paths": root_paths}),
    )

    while open_set:
        _, _, node = heapq.heappop(open_set)
        conflict = _detect_conflict(node["paths"])
        if conflict is None:
            return node["paths"]

        for conflicted_agent in (conflict["a1"], conflict["a2"]):
            child_constraints = list(node["constraints"])
            if conflict["kind"] == "vertex":
                child_constraints.append(
                    {
                        "agent": conflicted_agent,
                        "kind": "vertex",
                        "time": conflict["time"],
                        "cell": conflict["cell"],
                    }
                )
            else:
                from_cell, to_cell = conflict["from"], conflict["to"]
                if conflicted_agent == conflict["a2"]:
                    from_cell, to_cell = to_cell, from_cell
                child_constraints.append(
                    {
                        "agent": conflicted_agent,
                        "kind": "edge",
                        "time": conflict["time"],
                        "from": from_cell,
                        "to": to_cell,
                    }
                )

            child_paths = [list(path) for path in node["paths"]]
            replanned_path = _constrained_astar(
                binary_map,
                starts[conflicted_agent],
                goals[conflicted_agent],
                child_constraints,
                conflicted_agent,
                True,
            )
            if not replanned_path:
                continue

            child_paths[conflicted_agent] = replanned_path
            child_cost = sum(len(path) - 1 for path in child_paths)
            heapq.heappush(
                open_set,
                (
                    child_cost,
                    next(counter),
                    {"constraints": child_constraints, "paths": child_paths},
                ),
            )

    return []
