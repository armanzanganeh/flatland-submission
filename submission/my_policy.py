import numpy as np

class MyPolicy:
    def __init__(self):
        self.current_step = -1
        self.cached_actions = {}

    def act(self, handle, state, env, obs, info):
        step_id = env._elapsed_steps
        if step_id != self.current_step:
            self.current_step = step_id
            self.cached_actions = self._compute_all_actions(env)
        return self.cached_actions.get(handle, 4)

    def _compute_all_actions(self, env):
        actions = {}
        num_agents = env.get_num_agents()
        step_id = env._elapsed_steps
        move_vectors = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        
        # ۱. اولویت‌بندی قطارها بر اساس فاصله تا مقصد
        agent_priorities = []
        for h in range(num_agents):
            agent = env.agents[h]
            if agent.position is not None:
                dist = env.distance_map.get()[h, agent.position[0], agent.position[1], agent.direction]
            else:
                dist = 999999
            agent_priorities.append((dist, h))
        agent_priorities.sort()
        
        reserved_cells = set()
        current_positions = {env.agents[h].position for h in range(num_agents) if env.agents[h].position is not None}
        
        for dist, h in agent_priorities:
            agent = env.agents[h]
            
            # مدیریت ورود قطارها به نقشه
            if agent.position is None:
                if step_id > 0:
                    start_pos = agent.initial_position
                    if start_pos not in reserved_cells and start_pos not in current_positions:
                        actions[h] = 2
                        reserved_cells.add(start_pos)
                    else:
                        actions[h] = 4
                else:
                    actions[h] = 4
                continue

            pos = agent.position
            orientation = agent.direction
            
            transitions = env.rail.get_transitions((pos, orientation))
            action_to_dir = {1: (orientation - 1) % 4, 2: orientation, 3: (orientation + 1) % 4}
            
            # ۲. بررسی تمام حرکت‌های ممکن و رتبه‌بندی آن‌ها بر اساس فاصله
            possible_moves = []
            for action in [2, 1, 3]:  # اولویت با مستقیم، بعد چپ و راست
                next_dir = action_to_dir[action]
                if transitions[next_dir] == 1:
                    next_cell = (pos[0] + move_vectors[next_dir][0], pos[1] + move_vectors[next_dir][1])
                    d = env.distance_map.get()[h, next_cell[0], next_cell[1], next_dir]
                    possible_moves.append((d, action, next_cell))
            
            # مرتب‌سازی حرکت‌ها از بهترین (کمترین فاصله) به بدترین
            possible_moves.sort()
            
            # ۳. استراتژی فرار از ترافیک (Dynamic Rerouting)
            executed = False
            for d, action, next_cell in possible_moves:
                # اگر این مسیر (حتی اگر انتخاب اول نباشه) بازه و رزرو نشده، ازش برو!
                if next_cell not in reserved_cells:
                    actions[h] = action
                    reserved_cells.add(next_cell)
                    executed = True
                    break
            
            # اگر تمام مسیرهای ممکن رزرو شده بودند، حالا ناچاراً متوقف شو
            if not executed:
                actions[h] = 4
                reserved_cells.add(pos)
                
        return actions

    def step(self, handle, state, env, obs, info):
        return self.act(handle, state, env, obs, info)