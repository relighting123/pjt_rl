import numpy as np

class HeuristicExpert:
    def __init__(self, env):
        self.env = env
        self.num_prods = env.num_prods
        self.num_procs = env.num_procs

    def select_action(self, obs):
        # 1. State Extraction
        active_eqp = self.env.active_eqp
        target_eqp = self.env.target_eqp
        wip = self.env.wip
        st_map = self.env.st_map
        
        # 2. Calculate Priorities for all (Product, Process) pairs
        # We use 'Workload Hours' as the base priority
        priorities = np.zeros((self.num_prods, self.num_procs))
        
        for p in range(self.num_prods):
            for s in range(self.num_procs):
                prod_name = self.env.products[p]
                proc_name = self.env.processes[s]
                st = st_map.get((prod_name, proc_name), 999999)
                
                workload_hours = (wip[p, s] * st) / 60.0
                
                # Base Priority: Workload per equipment already allocated + 1 (for potential move)
                # This naturally balances equipment across processes
                current_allocation = active_eqp[p, s] + target_eqp[p, s]
                priority = workload_hours / (current_allocation + 1.0)
                
                # Boost for Final Step (Bottleneck Protection / Goal Achievement)
                if s == self.num_procs - 1 and wip[p, s] > 0:
                    priority *= 2.0  # Encourage finishing the product
                
                priorities[p, s] = priority

        # 3. Find the best target
        best_flat_idx = np.argmax(priorities)
        best_p = best_flat_idx // self.num_procs
        best_s = best_flat_idx % self.num_procs
        max_priority = priorities[best_p, best_s]
        
        # 4. Hysteresis & Move Decisions
        # To avoid ping-ponging, we define a significant gap needed to switch.
        # Also, we check if the current equipment is doing something important.
        
        # Find where we CAN move from (any station with active_eqp > 0)
        # We calculate the "loss" of moving from each occupied station.
        potential_sources = []
        for p in range(self.num_prods):
            for s in range(self.num_procs):
                if active_eqp[p, s] > 0:
                    # Current priority of this station (workload per allocated)
                    current_p = wip[p, s] * st_map.get((self.env.products[p], self.env.processes[s]), 999999) / 60.0
                    current_p /= (active_eqp[p, s] + target_eqp[p, s])
                    potential_sources.append((p, s, current_p))
        
        if not potential_sources:
            return best_flat_idx + 1 # No equipment anywhere? Just move to best.

        # Decide if moving is worth it
        # Logic: Move ONLY if (Best Target Priority) > (Current Source Priority) + Threshold
        # Threshold accounts for 1-hour changeover loss. 
        # For a move to be worth it, the target should have at least 2-3 hours of work.
        MOVE_THRESHOLD = 2.0 # Minimum priority gap to justify a move
        MIN_WORK_TO_MOVE = 3.0 # Hours of work required at target to justify a move
        
        best_source = min(potential_sources, key=lambda x: x[2]) # The station that needs its equipment the LEAST
        src_p, src_s, src_priority = best_source
        
        target_workload = (wip[best_p, best_s] * st_map.get((self.env.products[best_p], self.env.processes[best_s]), 999999)) / 60.0
        
        if max_priority > src_priority + MOVE_THRESHOLD and target_workload >= MIN_WORK_TO_MOVE:
            # Check if target is same as source (if so, stay)
            if best_p == src_p and best_s == src_s:
                return 0
            return best_flat_idx + 1
            
        return 0 # Stay put otherwise

    def generate_trajectories(self, num_episodes=10):
        data = []
        for _ in range(num_episodes):
            obs, _ = self.env.reset()
            done = False
            while not done:
                action = self.select_action(obs)
                next_obs, reward, terminated, truncated, _ = self.env.step(action)
                data.append((obs, action))
                obs = next_obs
                done = terminated or truncated
        return data
