import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from data_utils import DataLoader

class TaktEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, data_dir, max_steps=24, fixed_scenario=None):
        super(TaktEnv, self).__init__()
        self.data_dir = data_dir
        self.fixed_scenario = fixed_scenario
        self.scenarios = DataLoader.list_scenarios(data_dir)
        if not self.scenarios:
            raise ValueError(f"No scenarios found in {data_dir}")
        
        # Load first scenario as template for dimensions
        template_loader = DataLoader(data_dir, self.scenarios[0])
        self.max_steps = max_steps
        
        self.products = template_loader.get_products()
        self.processes = template_loader.get_processes()
        self.num_prods = len(self.products)
        self.num_procs = len(self.processes)

        # --- Data-driven global scaling factors (아이디어 1) ---
        # 모든 시나리오의 plan_wip / inventory / capabilities를 훑어서
        # 관측값들을 0~1 근처로 정규화하기 위한 스케일을 추정한다.
        self.max_wip_global = 0.0
        self.max_plan_global = 0.0
        self.max_eqp_global = 0.0
        self.max_st_global = 0.0
        for scn in self.scenarios:
            dl = DataLoader(self.data_dir, scn)
            # plan_wip 기반 WIP / PLAN 스케일
            for p in dl.plan_wip:
                self.max_wip_global = max(self.max_wip_global, float(p.get("wip", 0.0)))
                self.max_plan_global = max(self.max_plan_global, float(p.get("plan", 0.0)))
            # inventory 기반 설비 수 스케일 (총 설비 대수의 최대값 사용)
            total_eqp_scn = sum(item.get("count", 0.0) for item in dl.inventory)
            self.max_eqp_global = max(self.max_eqp_global, float(total_eqp_scn))
            # capabilities 기반 ST 스케일
            for cap in dl.capabilities:
                if cap.get("feasible", False):
                    self.max_st_global = max(self.max_st_global, float(cap.get("st", 0.0)))

        # 0으로 남아있으면 최소 1로 보정하여 나눗셈 안정화
        if self.max_wip_global <= 0.0:
            self.max_wip_global = 1.0
        if self.max_plan_global <= 0.0:
            self.max_plan_global = 1.0
        if self.max_eqp_global <= 0.0:
            self.max_eqp_global = 1.0
        if self.max_st_global <= 0.0:
            self.max_st_global = 60.0  # 기본 1시간 기준
        
        # Action space
        self.action_space = spaces.Discrete(self.num_prods * self.num_procs + 1)
        
        # Observation space: 
        # WIP(n), ActiveEqp(n), TargetEqp(n), CORemaining(n),
        # Achievement(n), ST(n), Plan(n), WIP/Plan Ratio(n),
        # TotalEqp(1), Step(1)
        # n = num_prods * num_procs
        self.obs_dim = (self.num_prods * self.num_procs * 8) + 2
        self.observation_space = spaces.Box(
            low=0, high=1000, 
            shape=(self.obs_dim,), 
            dtype=np.float32
        )

        # Cache ST map and other static data per scenario if needed, 
        # but for now we'll reload in reset for simplicity.
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        
        # Select scenario
        if options and "scenario" in options:
            self.current_scenario = options["scenario"]
        elif self.fixed_scenario:
            self.current_scenario = self.fixed_scenario
        else:
            # Randomly sample for training
            idx = self.np_random.integers(0, len(self.scenarios))
            self.current_scenario = self.scenarios[idx]
            
        self.loader = DataLoader(self.data_dir, self.current_scenario)
        self.st_map = self.loader.get_st_map()
        self.co_matrix, self.default_co = self.loader.get_changeover_matrix()
        self.total_eqp = self.loader.get_total_equipment()
        self.prod_idx = {p: i for i, p in enumerate(self.products)}
        self.proc_idx = {p: i for i, p in enumerate(self.processes)}
        
        wip_map, plan_map = self.loader.get_initial_wip_plan()
        
        # Initialize state
        self.wip = np.zeros((self.num_prods, self.num_procs))
        self.plan = np.zeros((self.num_prods, self.num_procs))
        self.produced = np.zeros((self.num_prods, self.num_procs))
        self.active_eqp = np.zeros((self.num_prods, self.num_procs))
        self.target_eqp = np.zeros((self.num_prods, self.num_procs))
        self.co_remaining = np.zeros((self.num_prods, self.num_procs))
        self.st_matrix = np.zeros((self.num_prods, self.num_procs))
        
        for (prod, proc), val in wip_map.items():
            if prod in self.prod_idx and proc in self.proc_idx:
                self.wip[self.prod_idx[prod], self.proc_idx[proc]] = val
        for (prod, proc), val in plan_map.items():
            if prod in self.prod_idx and proc in self.proc_idx:
                self.plan[self.prod_idx[prod], self.proc_idx[proc]] = val

        # Episode-level plan scale (for reward normalization, scenario 불문 동일 스케일)
        self.plan_total_last = float(np.sum(self.plan[:, -1]))
        self.plan_total_all = float(np.sum(self.plan))
        if self.plan_total_last <= 1e-6:
            self.plan_total_last = 1.0
        if self.plan_total_all <= 1e-6:
            self.plan_total_all = 1.0
        
        # Initialize ST matrix
        for (prod, proc), st in self.st_map.items():
            if prod in self.prod_idx and proc in self.proc_idx:
                self.st_matrix[self.prod_idx[prod], self.proc_idx[proc]] = st            
        # Initial equipment allocation
        for cap in self.loader.capabilities:
            if 'initial_count' in cap and cap['initial_count'] > 0:
                if cap['product'] in self.prod_idx and cap['process'] in self.proc_idx:
                    self.active_eqp[self.prod_idx[cap['product']], self.proc_idx[cap['process']]] += cap['initial_count']

        self.total_changeovers = 0
        self.history = []
        return self._get_obs(), {}

    def _get_obs(self):
        # Normalize values to improve generalization (아이디어 1 & 2 적용)
        # - WIP, Plan: 전 시나리오에서의 최대값으로 정규화
        # - Active/Target Eqp: 전 시나리오에서의 총 설비 대수 최대값으로 정규화
        # - ST: 전 시나리오에서의 ST 최대값으로 정규화
        # - Produced/Plan: 달성률
        # - WIP/Plan: 계획 대비 현재 잔여 WIP 비율 (아이디어 2)
        wip_norm = (self.wip / self.max_wip_global).flatten()
        active_norm = (self.active_eqp / self.max_eqp_global).flatten()
        target_norm = (self.target_eqp / self.max_eqp_global).flatten()
        co_norm = (self.co_remaining / 60.0).flatten()
        produced_ratio = (self.produced / (self.plan + 1e-6)).flatten()
        st_norm = (self.st_matrix / self.max_st_global).flatten()
        plan_norm = (self.plan / self.max_plan_global).flatten()
        wip_plan_ratio = (self.wip / (self.plan + 1e-6)).flatten()

        obs = np.concatenate([
            wip_norm,
            active_norm,
            target_norm,
            co_norm,
            produced_ratio,
            st_norm,
            plan_norm,
            wip_plan_ratio,
            [float(self.total_eqp) / self.max_eqp_global],
            [float(self.current_step) / float(self.max_steps)]
        ])
        return obs.astype(np.float32)

    def step(self, action):
        # 1. Handle Action (Equipment Reallocation)
        # Action is Move ONE equipment from an active/idle state to a new target
        if action > 0:
            target_idx = action - 1
            t_prod = target_idx // self.num_procs
            t_proc = target_idx % self.num_procs
            
            # Simple logic: Find an equipment to move. 
            # Prefer moving from somewhere with active_eqp > 0
            moved = False
            for p in range(self.num_prods):
                for s in range(self.num_procs):
                    if self.active_eqp[p, s] > 0 and (p != t_prod or s != t_proc):
                        self.active_eqp[p, s] -= 1
                        self.target_eqp[t_prod, t_proc] += 1
                        
                        # Calculate CO time
                        co_key = (self.products[p], self.processes[s], self.products[t_prod], self.processes[t_proc])
                        co_time = self.co_matrix.get(co_key, self.default_co)
                        # In this 1-hour step model, we'll round co_time in minutes to hours or handle it as fraction.
                        # User says 1 hour unit. Let's assume CO is in minutes.
                        # If CO > 0, it takes time.
                        self.co_remaining[t_prod, t_proc] = max(self.co_remaining[t_prod, t_proc], co_time / 60.0)
                        self.total_changeovers += 1
                        moved = True
                        break
                if moved: break

        # 2. Production Process (1 hour)
        hour_production = np.zeros_like(self.wip)
        for p in range(self.num_prods):
            for s in range(self.num_procs):
                prod_name = self.products[p]
                proc_name = self.processes[s]
                st = self.st_map.get((prod_name, proc_name), 999999)
                
                # Capacity: 60 mins / ST * active_eqp
                capacity = (60.0 / st) * self.active_eqp[p, s]
                # Available WIP
                actual_produce = min(capacity, self.wip[p, s])
                
                hour_production[p, s] = actual_produce
                self.produced[p, s] += actual_produce
                self.wip[p, s] -= actual_produce
                
                # WIP Flow to next process
                if s < self.num_procs - 1:
                    self.wip[p, s+1] += actual_produce

        # 3. Update Changeover
        for p in range(self.num_prods):
            for s in range(self.num_procs):
                if self.co_remaining[p, s] > 0:
                    self.co_remaining[p, s] -= 1.0 # 1 hour passed
                    if self.co_remaining[p, s] <= 0:
                        self.active_eqp[p, s] += self.target_eqp[p, s]
                        self.target_eqp[p, s] = 0
                        self.co_remaining[p, s] = 0

        # Logging
        for p in range(self.num_prods):
            for s in range(self.num_procs):
                self.history.append({
                    "timestamp": self.current_step,
                    "product": self.products[p],
                    "process": self.processes[s],
                    "wip": self.wip[p, s],
                    "production": hour_production[p, s],
                    "active_eqp": self.active_eqp[p, s],
                    "target_eqp": self.target_eqp[p, s],
                    "plan": self.plan[p, s],
                    "produced_sum": self.produced[p, s],
                    "total_changeovers": self.total_changeovers
                })

        self.current_step += 1
        terminated = self.current_step >= self.max_steps
        truncated = False
        
        # Reward: 시나리오 간 스케일을 맞추기 위해 PLAN 대비 비율로 정규화
        # - last_step_norm: 마지막 공정 생산량 / 에피소드 총 PLAN(마지막 공정)
        # - total_norm: 전체 생산량 / 에피소드 총 PLAN(전체 공정)
        last_step_norm = np.sum(hour_production[:, -1]) / (self.plan_total_last + 1e-6)
        total_norm = np.sum(hour_production) / (self.plan_total_all + 1e-6)

        # 계수(50, 20)는 시나리오가 달라도 보상 범위가 비슷하도록 맞추기 위한 경험적 값
        reward = last_step_norm * 50.0
        reward += total_norm * 20.0
        
        # Changeover Penalty: Encourage staying put
        if action > 0:
            reward -= 5.0 # Penalty per move
        
        # Bonus for meeting plan (기존과 동일, 이미 비율 보상)
        if terminated:
            final_achievement = np.sum(self.produced[:, -1]) / (np.sum(self.plan[:, -1]) + 1e-6)
            reward += final_achievement * 100
            
        return self._get_obs(), reward, terminated, truncated, {}

    def render(self):
        df = pd.DataFrame(self.history)
        if not df.empty:
            last_ts = df['timestamp'].max()
            print(f"--- Step {last_ts} ---")
            print(df[df['timestamp'] == last_ts][['product', 'process', 'wip', 'production', 'active_eqp', 'target_eqp']])

    def get_logs(self):
        return pd.DataFrame(self.history)
