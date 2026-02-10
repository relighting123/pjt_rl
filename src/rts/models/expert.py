import numpy as np
import logging

logger = logging.getLogger(__name__)

class HeuristicExpert:
    def __init__(self, env):
        self.env = env
        self.num_prods_padded = env.num_prods
        self.num_procs_padded = env.num_procs
        self.real_num_prods = env.real_num_prods
        self.real_num_procs = env.real_num_procs

    def select_action(self, obs):
        if self.env.active_eqp.ndim == 3:
            active_eqp = self.env.active_eqp.sum(axis=2)
            target_eqp = self.env.target_eqp.sum(axis=2)
        else:
            active_eqp = self.env.active_eqp
            target_eqp = self.env.target_eqp
        wip = self.env.wip
        st_map = self.env.st_map
        
        priorities = np.zeros((self.num_prods_padded, self.num_procs_padded))
        
        for p in range(self.real_num_prods):
            for s in range(self.real_num_procs):
                prod_name = self.env.products[p]
                proc_name = self.env.processes[s]
                
                # Calculate feasible allocation
                feasible_allocation = 0.0
                for m in range(self.env.num_models):
                    if self.env.st_matrix[p, s, m] > 0:
                        feasible_allocation += self.env.active_eqp[p, s, m] + self.env.target_eqp[p, s, m]
                
                # Min ST for this product/process
                sts = self.env.st_matrix[p, s, :]
                positive_sts = sts[sts > 0]
                min_st = positive_sts.min() if positive_sts.size > 0 else 999999
                
                workload_hours = (wip[p, s] * min_st) / 60.0
                priority = workload_hours / (feasible_allocation + 1.0)
                
                if s == self.real_num_procs - 1 and wip[p, s] > 0:
                    priority *= 2.0
                priorities[p, s] = priority

        best_flat_idx = np.argmax(priorities)
        best_p, best_s = best_flat_idx // self.num_procs_padded, best_flat_idx % self.num_procs_padded
        max_priority = priorities[best_p, best_s]
        
        potential_sources = []
        for p in range(self.real_num_prods):
            for s in range(self.real_num_procs):
                for m in range(self.env.num_models):
                    if self.env.active_eqp[p, s, m] > 0:
                        st_val = self.env.st_matrix[p, s, m]
                        if st_val > 0:
                            # Current priority of this feasible machine at its source
                            # Count how many feasible machines are at this source
                            source_feasible_alloc = sum(self.env.active_eqp[p, s, :] + self.env.target_eqp[p, s, :]) # simplifying
                            current_val = (wip[p, s] * st_val / 60.0) / (source_feasible_alloc)
                            potential_sources.append((p, s, m, current_val))
                        else:
                            # Non-feasible machine at this location (e.g. just moved there)
                            # Give it very low priority so it's moved first
                            potential_sources.append((p, s, m, -1.0))
        
        if not potential_sources:
            return best_flat_idx + 1

        MOVE_THRESHOLD = 2.0
        MIN_WORK_TO_MOVE = 3.0
        
        best_source = min(potential_sources, key=lambda x: x[3])
        src_p, src_s, src_m, src_priority = best_source
        st_val_target = st_map.get((self.env.products[best_p], self.env.processes[best_s]), 999999)
        target_workload = (wip[best_p, best_s] * st_val_target) / 60.0
        
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
