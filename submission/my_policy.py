from typing import Any, List, Dict
from heapq import heappush, heappop
from flatland.envs.rail_env_action import RailEnvActions


class MyPolicy:
    def __init__(self):
        self.res_table = {}  # (step, x, y) -> agent_handle
        self.paths = {}      # handle -> list of (x, y, direction)
        self.horizon = 10
        self.step_count = 0

    def act_many(self, handles: List[int], observations: List[Any], **kwargs) -> Dict[int, RailEnvActions]:
        env = observations[0] if observations and hasattr(observations[0], "agents") else None
        if env is None:
            return {h: RailEnvActions.MOVE_FORWARD for h in handles}
        self.step_count = env._elapsed_steps
        # پاک کردن رزرواسیون‌های قدیمی
        self.res_table = {k: v for k, v in self.res_table.items() if k[0] >= self.step_count}
        return self._step(env, handles)

    def _step(self, env, handles):
        actions = {}
        dist_map = env.distance_map.get()

        # اولویت‌بندی بر اساس فاصله تا مقصد
        priorities = []
        for h in handles:
            a = env.agents[h]
            if a.position is not None:
                x, y = a.position
                d = dist_map[h, x, y, a.direction]
            else:
                d = 999999
            priorities.append((d, h))
        priorities.sort()

        for _, h in priorities:
            agent = env.agents[h]
            state = agent.state if isinstance(agent.state, int) else agent.state.value

            if state in (5, 6, 7):
                actions[h] = RailEnvActions.DO_NOTHING
                continue

            # قطار هنوز spawn نشده
            if agent.position is None:
                start = agent.initial_position
                t = self.step_count
                if (t, start[0], start[1]) not in self.res_table:
                    actions[h] = RailEnvActions.MOVE_FORWARD
                    self.res_table[(t, start[0], start[1])] = h
                else:
                    actions[h] = RailEnvActions.DO_NOTHING
                continue

            # replan اگه مسیر نداره یا مسیر تموم شده
            if h not in self.paths or not self.paths[h]:
                self.paths[h] = self._astar(env, h)

            path = self.paths[h]

            if not path:
                actions[h] = RailEnvActions.DO_NOTHING
                continue

            next_node = path[0]
            nx, ny, nd = next_node
            t_next = self.step_count + 1

            # چک conflict با reservation table
            if (t_next, nx, ny) in self.res_table and self.res_table[(t_next, nx, ny)] != h:
                # replan با دور زدن conflict
                self.paths[h] = self._astar(env, h, blocked={(nx, ny)})
                if not self.paths[h]:
                    actions[h] = RailEnvActions.STOP_MOVING
                    continue
                next_node = self.paths[h][0]
                nx, ny, nd = next_node

            # رزرو مسیر
            for i, node in enumerate(path[:self.horizon]):
                sx, sy, _ = node
                self.res_table[(self.step_count + i + 1, sx, sy)] = h

            actions[h] = self._get_action(agent, nd)
            self.paths[h] = path[1:]

        return actions

    def _astar(self, env, h, blocked=set()):
        agent = env.agents[h]
        if agent.position is None:
            start = agent.initial_position
            start_dir = agent.initial_direction
        else:
            start = agent.position
            start_dir = agent.direction

        goal = agent.target
        if goal is None:
            return []

        dist_map = env.distance_map.get()
        move_vectors = [(-1,0),(0,1),(1,0),(0,-1)]

        def h_cost(x, y, d):
            try:
                return dist_map[h, x, y, d]
            except:
                return abs(x-goal[0]) + abs(y-goal[1])

        open_list = []
        heappush(open_list, (0, 0, start, start_dir, []))
        visited = set()
        counter = 0

        while open_list:
            f, g, (x, y), d, path = heappop(open_list)

            if (x, y) == goal:
                return path

            if (x, y, d) in visited:
                continue
            visited.add((x, y, d))

            cell = env.rail.grid[x, y]
            transitions = env.rail.transitions.get_transitions(cell, d)

            for nd in range(4):
                if not transitions[nd]:
                    continue
                dx, dy = move_vectors[nd]
                nx, ny = x+dx, y+dy

                if nx < 0 or ny < 0 or nx >= env.height or ny >= env.width:
                    continue
                if (nx, ny) in blocked:
                    continue

                new_g = g + 1
                counter += 1
                heappush(open_list, (new_g + h_cost(nx, ny, nd), new_g, (nx, ny), nd, path + [(nx, ny, nd)]))

        return []

    def _get_action(self, agent, next_dir):
        d = agent.direction
        if (d, next_dir) in [(0,3),(1,0),(2,1),(3,2)]:
            return RailEnvActions.MOVE_LEFT
        if (d, next_dir) in [(0,1),(1,2),(2,3),(3,0)]:
            return RailEnvActions.MOVE_RIGHT
        return RailEnvActions.MOVE_FORWARD

    def act(self, observation: Any, **kwargs) -> RailEnvActions:
        return RailEnvActions.MOVE_FORWARD