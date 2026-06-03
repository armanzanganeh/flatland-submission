from flatland.core.env_observation_builder import ObservationBuilder
from flatland.envs.rail_env_action import RailEnvActions

class MyObservationBuilder(ObservationBuilder):
    def __init__(self):
        super().__init__()

    def reset(self):
        pass

    def get(self, handle=0):
        return 0

    def get_many(self, handles=None):
        if handles is None:
            handles = []
            
        actions = {}
        move_vectors = [(-1,0),(0,1),(1,0),(0,-1)]
        env = self.env

        agent_priorities = []
        for h in handles:
            agent = env.agents[h]
            if agent.position is not None:
                dist = env.distance_map.get()[h, agent.position[0], agent.position[1], agent.direction]
            else:
                dist = 999999
            agent_priorities.append((dist, h))
        agent_priorities.sort()

        reserved_cells = set()
        current_positions = {env.agents[h].position for h in handles if env.agents[h].position is not None}

        for dist, h in agent_priorities:
            agent = env.agents[h]
            state = agent.state if isinstance(agent.state, int) else agent.state.value

            if state in (5, 6, 7):
                actions[h] = RailEnvActions.DO_NOTHING
                continue

            if agent.position is None:
                start_pos = agent.initial_position
                if start_pos not in reserved_cells and start_pos not in current_positions:
                    actions[h] = RailEnvActions.MOVE_FORWARD
                    reserved_cells.add(start_pos)
                else:
                    actions[h] = RailEnvActions.DO_NOTHING
                continue

            pos = agent.position
            direction = agent.direction
            
            # همون کد بهینه خودت
            cell_value = env.rail.grid[pos]
            transitions = env.rail.transitions.get_transitions(cell_value, direction)

            action_map = {
                (0,3): RailEnvActions.MOVE_LEFT,  (1,0): RailEnvActions.MOVE_LEFT,
                (2,1): RailEnvActions.MOVE_LEFT,  (3,2): RailEnvActions.MOVE_LEFT,
                (0,1): RailEnvActions.MOVE_RIGHT, (1,2): RailEnvActions.MOVE_RIGHT,
                (2,3): RailEnvActions.MOVE_RIGHT, (3,0): RailEnvActions.MOVE_RIGHT,
            }

            possible_moves = []
            for new_dir, allowed in enumerate(transitions):
                if not allowed:
                    continue
                next_cell = (pos[0]+move_vectors[new_dir][0], pos[1]+move_vectors[new_dir][1])
                d = env.distance_map.get()[h, next_cell[0], next_cell[1], new_dir]
                action = action_map.get((direction, new_dir), RailEnvActions.MOVE_FORWARD)
                possible_moves.append((d, action, next_cell))

            possible_moves.sort()

            executed = False
            for d, action, next_cell in possible_moves:
                if next_cell not in reserved_cells:
                    actions[h] = action
                    reserved_cells.add(next_cell)
                    executed = True
                    break

            if not executed:
                actions[h] = RailEnvActions.STOP_MOVING

        # اینجا اکشن‌ها رو به ترتیب شماره قطارها می‌چینیم تا توی شبکه قاطی نشن
        ordered_actions = {}
        for h in handles:
            ordered_actions[h] = actions.get(h, RailEnvActions.STOP_MOVING)
            
        return ordered_actions