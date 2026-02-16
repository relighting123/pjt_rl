import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import logging
from typing import Dict, Any
from ..data.data_loader import DataLoader
from ..config.config_manager import EnvConfig

logger = logging.getLogger(__name__)

class ProductionEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, data_dir: str = None, max_steps: int = None, fixed_scenario: str = None, config: EnvConfig = None, raw_data: Dict[str, Any] = None):
        super(ProductionEnv, self).__init__()
        self.config = config or EnvConfig()
        self.data_dir = data_dir
        self.fixed_scenario = fixed_scenario
        self.raw_data = raw_data
        
        # Discovery of all unique entities across all scenarios
        self.products = set()
        self.processes = set()
        self.models = set()

        if self.raw_data:
            # DB-direct mode: discover entities only from the provided data
            self.scenarios = [fixed_scenario or "raw_data"]
            self.loader = DataLoader(raw_data=self.raw_data)
            self.products.update(self.loader.get_products())
            self.processes.update(self.loader.get_processes())
            self.models.update(self.loader.get_models())
            logger.info("Initializing ProductionEnv with provided raw_data")
        elif self.fixed_scenario:
            # Single-scenario file mode (inference): discover entities only from the target scenario
            # This ensures consistent behavior with the DB-direct path for the same data
            self.scenarios = DataLoader.list_scenarios(data_dir)
            if not self.scenarios:
                raise ValueError(f"No scenarios found in {data_dir}")
            
            logger.info(f"Initializing ProductionEnv with fixed scenario '{self.fixed_scenario}' from {data_dir}")
            
            dl = DataLoader(data_dir, self.fixed_scenario)
            self.products.update(dl.get_products())
            self.processes.update(dl.get_processes())
            self.models.update(dl.get_models())
        else:
            # Multi-scenario mode (training): discover entities from ALL scenarios
            self.scenarios = DataLoader.list_scenarios(data_dir)
            if not self.scenarios:
                raise ValueError(f"No scenarios found in {data_dir}")
            
            logger.info(f"Initializing ProductionEnv with {len(self.scenarios)} scenarios from {data_dir}")
            
            for scn in self.scenarios:
                dl = DataLoader(data_dir, scn)
                self.products.update(dl.get_products())
                self.processes.update(dl.get_processes())
                self.models.update(dl.get_models())
            
        self.max_steps = max_steps or self.config.max_steps
            
        self.products = sorted(list(self.products))
        self.processes = sorted(list(self.processes))
        self.models = sorted(list(self.models))

        self.num_prods = self.config.max_prods
        self.num_procs = self.config.max_procs
        
        self.real_num_prods = min(len(self.products), self.num_prods)
        self.real_num_procs = min(len(self.processes), self.num_procs)

        # Pad name lists to match max dimensions
        while len(self.products) < self.num_prods:
            self.products.append(f"PAD_PROD_{len(self.products)}")
        while len(self.processes) < self.num_procs:
            self.processes.append(f"PAD_PROC_{len(self.processes)}")
            
        self.num_models = len(self.models) # num_models is used internally, observation is aggregated

        # Global scaling factors estimation
        self._estimate_global_scales()
        
        # Action space: (max_prods * max_procs) + 1 (idle)
        self.action_space = spaces.Discrete(self.num_prods * self.num_procs + 1)
        
        # Observation space
        self.obs_dim = (self.num_prods * self.num_procs * 8) + 2
        self.observation_space = spaces.Box(
            low=0, high=1000, 
            shape=(self.obs_dim,), 
            dtype=np.float32
        )

        self.reset()

    def _estimate_global_scales(self):
        self.max_wip_global = 0.0
        self.max_plan_global = 0.0
        self.max_eqp_global = 0.0
        self.max_st_global = 0.0
        
        if self.raw_data:
            # DB-direct mode: estimate scales from the provided data only
            for p in self.loader.plan_wip:
                self.max_wip_global = max(self.max_wip_global, float(p.wip))
                self.max_plan_global = max(self.max_plan_global, float(p.plan))
            
            total_eqp_scn = self.loader.get_total_equipment()
            self.max_eqp_global = max(self.max_eqp_global, float(total_eqp_scn))
            
            for cap in self.loader.capabilities:
                if cap.feasible:
                    self.max_st_global = max(self.max_st_global, float(cap.st))
        elif self.fixed_scenario:
            # Single-scenario file mode (inference): estimate scales from the target scenario only
            # This ensures consistent normalization with the DB-direct path for the same data
            try:
                dl = DataLoader(self.data_dir, self.fixed_scenario)
                for p in dl.plan_wip:
                    self.max_wip_global = max(self.max_wip_global, float(p.wip))
                    self.max_plan_global = max(self.max_plan_global, float(p.plan))
                
                total_eqp_scn = dl.get_total_equipment()
                self.max_eqp_global = max(self.max_eqp_global, float(total_eqp_scn))
                
                for cap in dl.capabilities:
                    if cap.feasible:
                        self.max_st_global = max(self.max_st_global, float(cap.st))
            except Exception as e:
                logger.warning(f"Failed to scan fixed scenario {self.fixed_scenario} for scales: {e}")
        else:
            # Multi-scenario mode (training): estimate scales from ALL scenarios
            for scn in self.scenarios:
                try:
                    dl = DataLoader(self.data_dir, scn)
                    for p in dl.plan_wip:
                        self.max_wip_global = max(self.max_wip_global, float(p.wip))
                        self.max_plan_global = max(self.max_plan_global, float(p.plan))
                    
                    total_eqp_scn = dl.get_total_equipment()
                    self.max_eqp_global = max(self.max_eqp_global, float(total_eqp_scn))
                    
                    for cap in dl.capabilities:
                        if cap.feasible:
                            self.max_st_global = max(self.max_st_global, float(cap.st))
                except Exception as e:
                    logger.warning(f"Failed to scan scenario {scn} for scales: {e}")

        # Basic stabilization
        self.max_wip_global = max(self.max_wip_global, 1.0)
        self.max_plan_global = max(self.max_plan_global, 1.0)
        self.max_eqp_global = max(self.max_eqp_global, 1.0)
        self.max_st_global = max(self.max_st_global, 60.0)
        
        logger.debug(f"Global scales - WIP: {self.max_wip_global}, Plan: {self.max_plan_global}, Eqp: {self.max_eqp_global}, ST: {self.max_st_global}")

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        
        if self.raw_data:
            self.current_scenario = self.fixed_scenario or "raw_data"
        elif options and "scenario" in options:
            self.current_scenario = options["scenario"]
            self.loader = DataLoader(self.data_dir, self.current_scenario)
        elif self.fixed_scenario:
            self.current_scenario = self.fixed_scenario
            self.loader = DataLoader(self.data_dir, self.current_scenario)
        else:
            idx = self.np_random.integers(0, len(self.scenarios))
            self.current_scenario = self.scenarios[idx]
            self.loader = DataLoader(self.data_dir, self.current_scenario)
        self.st_map = self.loader.get_st_map()
        self.co_matrix, self.default_co = self.loader.get_changeover_matrix()
        self.total_eqp = self.loader.get_total_equipment()
        self.prod_idx = {p: i for i, p in enumerate(self.products) if i < self.num_prods}
        self.proc_idx = {p: i for i, p in enumerate(self.processes) if i < self.num_procs}
        self.model_idx = {m: i for i, m in enumerate(self.models)}
        self.batch_map = self.loader.get_batch_map()
        self.tool_inventory = self.loader.get_tool_inventory()
        
        logger.debug(f"Resetting Env - Scenario: {self.current_scenario}")
        logger.debug(f"Padded Prods: {self.num_prods}, Real Prods: {self.real_num_prods}")
        logger.debug(f"Padded Procs: {self.num_procs}, Real Procs: {self.real_num_procs}")
        
        wip_map, plan_map = self.loader.get_initial_wip_plan()
        
        self.wip = np.zeros((self.num_prods, self.num_procs))
        self.plan = np.zeros((self.num_prods, self.num_procs))
        self.produced = np.zeros((self.num_prods, self.num_procs))

        self.active_eqp = np.zeros((self.num_prods, self.num_procs, self.num_models))
        self.target_eqp = np.zeros((self.num_prods, self.num_procs, self.num_models))
        self.co_remaining = np.zeros((self.num_prods, self.num_procs, self.num_models))
        self.st_matrix = np.zeros((self.num_prods, self.num_procs, self.num_models))
        
        for (prod, proc), val in wip_map.items():
            if prod in self.prod_idx and proc in self.proc_idx:
                self.wip[self.prod_idx[prod], self.proc_idx[proc]] = val
        for (prod, proc), val in plan_map.items():
            if prod in self.prod_idx and proc in self.proc_idx:
                self.plan[self.prod_idx[prod], self.proc_idx[proc]] = val

        # Track the actual last process index for this scenario
        current_scenario_procs = self.loader.get_processes()
        if current_scenario_procs:
            last_proc_name = current_scenario_procs[-1]
            self.last_proc_idx_current = self.proc_idx.get(last_proc_name, self.real_num_procs - 1)
        else:
            self.last_proc_idx_current = self.real_num_procs - 1

        self.plan_total_last = float(np.sum(self.plan[:, self.last_proc_idx_current]))
        self.plan_total_all = float(np.sum(self.plan))
        if self.plan_total_last <= 1e-6: self.plan_total_last = 1.0
        if self.plan_total_all <= 1e-6: self.plan_total_all = 1.0
        
        for cap in self.loader.capabilities:
            if cap.feasible:
                prod, proc, model, st_val = cap.product, cap.process, cap.model, cap.st
                if prod in self.prod_idx and proc in self.proc_idx and model in self.model_idx:
                    i, j, k = self.prod_idx[prod], self.proc_idx[proc], self.model_idx[model]
                    self.st_matrix[i, j, k] = st_val

        for cap in self.loader.capabilities:
            if cap.initial_count > 0:
                prod, proc, model = cap.product, cap.process, cap.model
                if prod in self.prod_idx and proc in self.proc_idx and model in self.model_idx:
                    i, j, k = self.prod_idx[prod], self.proc_idx[proc], self.model_idx[model]
                    self.active_eqp[i, j, k] += cap.initial_count

        # 4. Initialize Idle Pool
        self.idle_eqp = np.zeros(self.num_models)
        inv_counts = {item.model: item.count for item in self.loader.inventory}
        for m_name, total_count in inv_counts.items():
            if m_name in self.model_idx:
                k = self.model_idx[m_name]
                assigned_count = self.active_eqp[:, :, k].sum()
                self.idle_eqp[k] = max(0, total_count - assigned_count)

        # Initialize Tool Usage
        self.tool_usage = {b: 0 for b in self.tool_inventory.keys()}
        for p in range(self.num_prods):
            for s in range(self.num_procs):
                batch = self.batch_map.get((self.products[p], self.processes[s]))
                if batch and batch in self.tool_usage:
                    self.tool_usage[batch] += self.active_eqp[p, s, :].sum() + self.target_eqp[p, s, :].sum()

        self.total_changeovers = 0
        self.history = []

        # Tracking for log filtering
        self.current_scenario_products = set(self.loader.get_products())
        self.current_scenario_processes = set(self.loader.get_processes())

        # Downtime tracking
        self.downtime_schedule = self.loader.downtime
        
        return self._get_obs(), {}

    def _get_obs(self):
        wip_norm = (self.wip / self.max_wip_global).flatten()

        active_total = self.active_eqp.sum(axis=2)
        target_total = self.target_eqp.sum(axis=2)
        if self.num_models > 0:
            co_total = self.co_remaining.max(axis=2)
        else:
            co_total = np.zeros((self.num_prods, self.num_procs))

        active_norm = (active_total / self.max_eqp_global).flatten()
        target_norm = (target_total / self.max_eqp_global).flatten()
        co_norm = (co_total / 60.0).flatten()
        produced_ratio = (self.produced / (self.plan + 1e-6)).flatten()
        
        st_per_pp = np.zeros((self.num_prods, self.num_procs))
        for i in range(self.num_prods):
            for j in range(self.num_procs):
                sts = self.st_matrix[i, j, :]
                positive_sts = sts[sts > 0]
                st_val = positive_sts.min() if positive_sts.size > 0 else 0.0
                st_per_pp[i, j] = st_val
        st_norm = (st_per_pp / self.max_st_global).flatten()
        plan_norm = (self.plan / self.max_plan_global).flatten()
        wip_plan_ratio = (self.wip / (self.plan + 1e-6)).flatten()

        obs = np.concatenate([
            wip_norm, active_norm, target_norm, co_norm,
            produced_ratio, st_norm, plan_norm, wip_plan_ratio,
            [float(self.total_eqp) / self.max_eqp_global],
            [float(self.current_step) / float(self.max_steps)]
        ])
        return obs.astype(np.float32)

    def step(self, action):
        if action > 0:
            target_idx = action - 1
            t_prod, t_proc = target_idx // self.num_procs, target_idx % self.num_procs
            
            # Find feasible models for target (st > 0)
            feasible_models = [m for m in range(self.num_models) if self.st_matrix[t_prod, t_proc, m] > 0]
            
            moved = False
            src_info = None # (p, s, m) - if s is -1, it's from idle pool
            
            # 1. Try to find a feasible model in the IDLE POOL first
            target_batch = self.batch_map.get((self.products[t_prod], self.processes[t_proc]))
            
            for m in feasible_models:
                if self.idle_eqp[m] > 0:
                    # Check tool availability if target has a batch constraint
                    if target_batch in self.tool_inventory:
                        if self.tool_usage.get(target_batch, 0) >= self.tool_inventory[target_batch]:
                            continue # Tool not available
                    
                    self.idle_eqp[m] -= 1
                    self.target_eqp[t_prod, t_proc, m] += 1
                    if target_batch in self.tool_usage:
                        self.tool_usage[target_batch] += 1
                        
                    # Changeover from idle is assumed to be default_co or 0? 
                    # Let's assume 0 for simplicity if coming from idle, or default_co if not
                    self.co_remaining[t_prod, t_proc, m] = self.default_co / 60.0 
                    self.total_changeovers += 1
                    moved = True
                    break
            
            # 2. Try to find a feasible model at another ACTIVE location
            if not moved:
                for m in feasible_models:
                    for p in range(self.num_prods):
                        for s in range(self.num_procs):
                            if self.active_eqp[p, s, m] > 0 and (p != t_prod or s != t_proc):
                                src_info = (p, s, m)
                                moved = True
                                break
                        if moved: break
                    if moved: break

                if moved and src_info is not None:
                    sp, ss, sm = src_info
                    source_batch = self.batch_map.get((self.products[sp], self.processes[ss]))
                    
                    # Check tool availability if batch changes
                    if source_batch != target_batch:
                        if target_batch in self.tool_inventory:
                            if self.tool_usage.get(target_batch, 0) >= self.tool_inventory[target_batch]:
                                moved = False # Tool not available, move canceled
                    
                    if moved:
                        self.active_eqp[sp, ss, sm] -= 1
                        self.target_eqp[t_prod, t_proc, sm] += 1
                        
                        if source_batch != target_batch:
                            if source_batch in self.tool_usage:
                                self.tool_usage[source_batch] -= 1
                            if target_batch in self.tool_usage:
                                self.tool_usage[target_batch] += 1

                        co_key = (self.products[sp], self.processes[ss], self.products[t_prod], self.processes[t_proc])
                        co_time = self.co_matrix.get(co_key, self.default_co)
                        self.co_remaining[t_prod, t_proc, sm] = max(self.co_remaining[t_prod, t_proc, sm], co_time / 60.0)
                        self.total_changeovers += 1

        hour_production = np.zeros_like(self.wip)
        
        # Calculate currently unavailable equipment due to downtime
        unavailable_eqp = np.zeros(self.num_models)
        for dt in self.downtime_schedule:
            if dt.start_step <= self.current_step < dt.end_step:
                if dt.model in self.model_idx:
                    unavailable_eqp[self.model_idx[dt.model]] += dt.count

        for p in range(self.num_prods):
            for s in range(self.num_procs):
                capacity = 0.0
                for m in range(self.num_models):
                    st_val = self.st_matrix[p, s, m]
                    if st_val > 0.0:
                        # Effective active equipment = active_eqp - pro-rated downtime
                        # This is a simplified logic: we assume downtime reduces total pool capacity proportionally 
                        # Or more accurately, if active_eqp[p,s,m] is 2 and total of model m is 10, 
                        # and 5 are down, then effectively at most 5 are available across the whole factory.
                        # For simplicity, we just reduce the capacity globally or per-cell if we had machine-specific IDs.
                        # Since we only have counts, we'll subtract unavailable count from total active count for that model.
                        
                        # Total active of model 'm' across factory
                        total_active_m = self.active_eqp[:, :, m].sum()
                        if total_active_m > 0:
                            # Percentage of model 'm' that is actually available
                            effective_ratio = max(0.0, (total_active_m - unavailable_eqp[m]) / total_active_m)
                            capacity += (60.0 / st_val) * (self.active_eqp[p, s, m] * effective_ratio)
                        else:
                            capacity += 0.0

                actual_produce = min(capacity, self.wip[p, s])
                hour_production[p, s] = actual_produce
                self.produced[p, s] += actual_produce
                self.wip[p, s] -= actual_produce
                if s < self.num_procs - 1:
                    self.wip[p, s+1] += actual_produce

        for p in range(self.num_prods):
            for s in range(self.num_procs):
                for m in range(self.num_models):
                    if self.co_remaining[p, s, m] > 0:
                        self.co_remaining[p, s, m] -= 1.0
                        if self.co_remaining[p, s, m] <= 0:
                            self.active_eqp[p, s, m] += self.target_eqp[p, s, m]
                            self.target_eqp[p, s, m] = 0
                            self.co_remaining[p, s, m] = 0

        for p in range(self.num_prods):
            # Skip logging for entities not in the current scenario
            if self.products[p] not in self.current_scenario_products:
                continue
                
            for s in range(self.num_procs):
                if self.processes[s] not in self.current_scenario_processes:
                    continue
                    
                self.history.append({
                    "timestamp": self.current_step,
                    "product": self.products[p],
                    "process": self.processes[s],
                    "wip": self.wip[p, s],
                    "production": hour_production[p, s],
                    "active_eqp": float(self.active_eqp[p, s, :].sum()),
                    "target_eqp": float(self.target_eqp[p, s, :].sum()),
                    "unavailable_eqp": float(unavailable_eqp.sum()) if p == 0 and s == 0 else 0.0, # Log once per step
                    "plan": self.plan[p, s],
                    "produced_sum": self.produced[p, s],
                    "total_changeovers": self.total_changeovers
                })

        self.current_step += 1
        terminated = self.current_step >= self.max_steps
        truncated = False
        
        last_step_norm = np.sum(hour_production[:, self.last_proc_idx_current]) / (self.plan_total_last + 1e-6)
        total_norm = np.sum(hour_production) / (self.plan_total_all + 1e-6)

        reward = last_step_norm * self.config.reward.last_step_weight
        reward += total_norm * self.config.reward.total_production_weight
        if action > 0:
            reward -= self.config.reward.changeover_penalty
        
        if terminated:
            final_achievement = np.sum(self.produced[:, -1]) / (np.sum(self.plan[:, -1]) + 1e-6)
            reward += final_achievement * self.config.reward.final_achievement_bonus
            
        return self._get_obs(), reward, terminated, truncated, {}

    def render(self):
        df = pd.DataFrame(self.history)
        if not df.empty:
            last_ts = df['timestamp'].max()
            logger.info(f"--- Step {last_ts} ---")
            # For render, we might still want to print or use a specific logger handler
            print(df[df['timestamp'] == last_ts][['product', 'process', 'wip', 'production', 'active_eqp', 'target_eqp']])

    def get_logs(self):
        return pd.DataFrame(self.history)
