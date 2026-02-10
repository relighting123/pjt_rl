import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import logging
from ..data.data_loader import DataLoader
from ..config.config_manager import EnvConfig

logger = logging.getLogger(__name__)

class TaktEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, data_dir, max_steps=None, fixed_scenario=None, config: EnvConfig = None):
        super(TaktEnv, self).__init__()
        self.config = config or EnvConfig()
        self.data_dir = data_dir
        self.fixed_scenario = fixed_scenario
        self.scenarios = DataLoader.list_scenarios(data_dir)
        if not self.scenarios:
            raise ValueError(f"No scenarios found in {data_dir}")
        
        logger.info(f"Initializing TaktEnv with {len(self.scenarios)} scenarios from {data_dir}")
        
        # Load first scenario as template for dimensions
        template_loader = DataLoader(data_dir, self.scenarios[0])
        self.max_steps = max_steps or self.config.max_steps
        
        self.products = template_loader.get_products()
        self.processes = template_loader.get_processes()
        self.models = template_loader.get_models()

        self.num_prods = len(self.products)
        self.num_procs = len(self.processes)
        self.num_models = len(self.models)

        # Global scaling factors estimation
        self._estimate_global_scales()
        
        # Action space
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
        
        if options and "scenario" in options:
            self.current_scenario = options["scenario"]
        elif self.fixed_scenario:
            self.current_scenario = self.fixed_scenario
        else:
            idx = self.np_random.integers(0, len(self.scenarios))
            self.current_scenario = self.scenarios[idx]
            
        self.loader = DataLoader(self.data_dir, self.current_scenario)
        self.st_map = self.loader.get_st_map()
        self.co_matrix, self.default_co = self.loader.get_changeover_matrix()
        self.total_eqp = self.loader.get_total_equipment()
        self.prod_idx = {p: i for i, p in enumerate(self.products)}
        self.proc_idx = {p: i for i, p in enumerate(self.processes)}
        self.model_idx = {m: i for i, m in enumerate(self.models)}
        
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

        self.plan_total_last = float(np.sum(self.plan[:, -1]))
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

        self.total_changeovers = 0
        self.history = []
        return self._get_obs(), {}

    def _get_obs(self):
        wip_norm = (self.wip / self.max_wip_global).flatten()

        active_total = self.active_eqp.sum(axis=2)
        target_total = self.target_eqp.sum(axis=2)
        co_total = self.co_remaining.max(axis=2)

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
            
            moved = False
            src_info = None
            for p in range(self.num_prods):
                for s in range(self.num_procs):
                    for m in range(self.num_models):
                        if self.active_eqp[p, s, m] > 0 and (p != t_prod or s != t_proc):
                            src_info = (p, s, m)
                            moved = True
                            break
                    if moved: break
                if moved: break

            if moved and src_info is not None:
                sp, ss, sm = src_info
                self.active_eqp[sp, ss, sm] -= 1
                self.target_eqp[t_prod, t_proc, sm] += 1

                co_key = (self.products[sp], self.processes[ss], self.products[t_prod], self.processes[t_proc])
                co_time = self.co_matrix.get(co_key, self.default_co)
                self.co_remaining[t_prod, t_proc, sm] = max(self.co_remaining[t_prod, t_proc, sm], co_time / 60.0)
                self.total_changeovers += 1

        hour_production = np.zeros_like(self.wip)
        for p in range(self.num_prods):
            for s in range(self.num_procs):
                capacity = 0.0
                for m in range(self.num_models):
                    st_val = self.st_matrix[p, s, m]
                    if st_val > 0.0:
                        capacity += (60.0 / st_val) * self.active_eqp[p, s, m]

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
            for s in range(self.num_procs):
                self.history.append({
                    "timestamp": self.current_step,
                    "product": self.products[p],
                    "process": self.processes[s],
                    "wip": self.wip[p, s],
                    "production": hour_production[p, s],
                    "active_eqp": float(self.active_eqp[p, s, :].sum()),
                    "target_eqp": float(self.target_eqp[p, s, :].sum()),
                    "plan": self.plan[p, s],
                    "produced_sum": self.produced[p, s],
                    "total_changeovers": self.total_changeovers
                })

        self.current_step += 1
        terminated = self.current_step >= self.max_steps
        truncated = False
        
        last_step_norm = np.sum(hour_production[:, -1]) / (self.plan_total_last + 1e-6)
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
