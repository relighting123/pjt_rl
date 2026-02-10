import numpy as np
import logging

logger = logging.getLogger(__name__)

class HeuristicExpert:
    def __init__(self, env):
        self.env = env
        self.num_prods = env.num_prods
        self.num_procs = env.num_procs

    def select_action(self, obs):
        if self.env.active_eqp.ndim == 3:
            active_eqp = self.env.active_eqp.sum(axis=2)
            target_eqp = self.env.target_eqp.sum(axis=2)
        else:
            active_eqp = self.env.active_eqp
            target_eqp = self.env.target_eqp
        wip = self.env.wip
        st_map = self.env.st_map
        
        priorities = np.zeros((self.num_prods, self.num_procs))
        
        for p in range(self.num_prods):
            for s in range(self.num_procs):
                prod_name = self.env.products[p]
                proc_name = self.env.processes[s]
                st = st_map.get((prod_name, proc_name), 999999)
                workload_hours = (wip[p, s] * st) / 60.0
                current_allocation = active_eqp[p, s] + target_eqp[p, s]
                priority = workload_hours / (current_allocation + 1.0)
                if s == self.num_procs - 1 and wip[p, s] > 0:
                    priority *= 2.0
                priorities[p, s] = priority

        best_flat_idx = np.argmax(priorities)
        best_p, best_s = best_flat_idx // self.num_procs, best_flat_idx % self.num_procs
        max_priority = priorities[best_p, best_s]
        
        potential_sources = []
        for p in range(self.num_prods):
            for s in range(self.num_procs):
                if active_eqp[p, s] > 0:
                    current_p = wip[p, s] * st_map.get((self.env.products[p], self.env.processes[s]), 999999) / 60.0
                    current_p /= (active_eqp[p, s] + target_eqp[p, s])
                    potential_sources.append((p, s, current_p))
        
        if not potential_sources:
            return best_flat_idx + 1

        MOVE_THRESHOLD = 2.0
        MIN_WORK_TO_MOVE = 3.0
        
        best_source = min(potential_sources, key=lambda x: x[2])
        src_p, src_s, src_priority = best_source
        target_workload = (wip[best_p, best_s] * st_map.get((self.env.products[best_p], self.env.processes[best_s]), 999999)) / 60.0
        
        if max_priority > src_priority + MOVE_THRESHOLD and target_workload >= MIN_WORK_TO_MOVE:
            if best_p == src_p and best_s == src_s:
                return 0
            return best_flat_idx + 1
            
        return 0

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
