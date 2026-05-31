import heapq
from typing import Any, List, Dict
from flatland.envs.rail_env_action import RailEnvActions
from flatland.envs.RailEnvPolicy import RailEnvPolicy


def a_star_search(env, start_pos, start_dir, target_pos, blocked=set()):
    open_set = []
    counter = 0
    heapq.heappush(open_set, (0, 0, counter, tuple(start_pos), start_dir, []))
    visited = set()
    while open_set:
        _, g, _, pos, direction, path = heapq.heappop(open_set)
        if pos == tuple(target_pos):
            return path
        if (pos, direction) in visited:
            continue
        visited.add((pos, direction))
        cell_value = env.rail.grid[pos]
        transitions = env.rail.transitions.get_transitions(cell_value, direction)
        for new_dir, allowed in enumerate(transitions):
            if not allowed:
                continue
            delta = [(-1,0),(0,1),(1,0),(0,-1)]
            next_pos = (pos[0]+delta[new_dir][0], pos[1]+delta[new_dir][1])
            if not (0 <= next_pos[0] < env.height and 0 <= next_pos[1] < env.width):
                continue
            if next_pos in blocked:
                continue
            action_map = {
                (0,3): RailEnvActions.MOVE_LEFT,  (1,0): RailEnvActions.MOVE_LEFT,
                (2,1): RailEnvActions.MOVE_LEFT,  (3,2): RailEnvActions.MOVE_LEFT,
                (0,1): RailEnvActions.MOVE_RIGHT, (1,2): RailEnvActions.MOVE_RIGHT,
                (2,3): RailEnvActions.MOVE_RIGHT, (3,0): RailEnvActions.MOVE_RIGHT,
            }
            action = action_map.get((direction, new_dir), RailEnvActions.MOVE_FORWARD)
            h = abs(next_pos[0]-target_pos[0]) + abs(next_pos[1]-target_pos[1])
            counter += 1
            heapq.heappush(open_set, (g+1+h, g+1, counter, next_pos, new_dir, path+[action]))
    return None


def get_priority(agent):
    if agent.position is None:
        return float('inf')
    return abs(agent.position[0]-agent.target[0]) + abs(agent.position[1]-agent.target[1])


class MyPolicy(RailEnvPolicy):
    def __init__(self):
        self.agent_paths = {}
        self.last_positions = {}
        self.stuck_counter = {}
        self.STUCK_LIMIT = 3

    def act_many(self, handles: List[int], observations: List[Any], **kwargs) -> Dict[int, RailEnvActions]:
        env = observations[0] if observations and hasattr(observations[0], 'agents') else None
        if env is None:
            return {a: RailEnvActions.MOVE_FORWARD for a in handles}

        actions = {}
        occupied = {tuple(ag.position) for ag in env.agents if ag.position is not None}
        sorted_handles = sorted(handles, key=lambda a: get_priority(env.agents[a]))
        reserved_next = set()

        for a in sorted_handles:
            agent = env.agents[a]
            state = agent.state if isinstance(agent.state, int) else agent.state.value

            if state in (0, 5, 6, 7):
                actions[a] = RailEnvActions.DO_NOTHING
                continue
            if state == 1:
                actions[a] = RailEnvActions.MOVE_FORWARD
                continue

            pos = tuple(agent.position)

            if self.last_positions.get(a) == pos:
                self.stuck_counter[a] = self.stuck_counter.get(a, 0) + 1
            else:
                self.stuck_counter[a] = 0
            self.last_positions[a] = pos

            if self.stuck_counter.get(a, 0) >= self.STUCK_LIMIT or a not in self.agent_paths or not self.agent_paths[a]:
                blocked = occupied - {pos}
                path = a_star_search(env, pos, agent.direction, tuple(agent.target), blocked)
                if not path:
                    path = a_star_search(env, pos, agent.direction, tuple(agent.target))
                self.agent_paths[a] = path or []
                self.stuck_counter[a] = 0

            if self.agent_paths.get(a):
                next_action = self.agent_paths[a][0]
                delta = [(-1,0),(0,1),(1,0),(0,-1)]
                transitions = env.rail.transitions.get_transitions(
                    env.rail.grid[pos], agent.direction
                )
                next_pos = pos
                for new_dir, allowed in enumerate(transitions):
                    if not allowed:
                        continue
                    candidate = (pos[0]+delta[new_dir][0], pos[1]+delta[new_dir][1])
                    action_map = {
                        (0,3): RailEnvActions.MOVE_LEFT,  (1,0): RailEnvActions.MOVE_LEFT,
                        (2,1): RailEnvActions.MOVE_LEFT,  (3,2): RailEnvActions.MOVE_LEFT,
                        (0,1): RailEnvActions.MOVE_RIGHT, (1,2): RailEnvActions.MOVE_RIGHT,
                        (2,3): RailEnvActions.MOVE_RIGHT, (3,0): RailEnvActions.MOVE_RIGHT,
                    }
                    if action_map.get((agent.direction, new_dir), RailEnvActions.MOVE_FORWARD) == next_action:
                        next_pos = candidate
                        break

                if next_pos in reserved_next:
                    actions[a] = RailEnvActions.STOP_MOVING
                else:
                    reserved_next.add(next_pos)
                    actions[a] = self.agent_paths[a].pop(0)
            else:
                actions[a] = RailEnvActions.MOVE_FORWARD

        return actions

    def act(self, observation: Any, **kwargs) -> RailEnvActions:
        return RailEnvActions.MOVE_FORWARD