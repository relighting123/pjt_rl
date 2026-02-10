import json
import os
import logging
from typing import List, Dict, Any, Tuple
from pydantic import BaseModel, Field, ValidationError

logger = logging.getLogger(__name__)

# --- Pydantic Schemas for Validation ---

class ChangeoverRule(BaseModel):
    from_product: str
    from_process: str
    to_product: str
    to_process: str
    time: float

class ChangeoverRules(BaseModel):
    default_time: float
    rules: List[ChangeoverRule]

class Capability(BaseModel):
    product: str
    process: str
    model: str
    st: float
    feasible: bool
    initial_count: int = 0

class InventoryItem(BaseModel):
    model: str
    count: int

class PlanWIPItem(BaseModel):
    product: str
    process: str
    oper_seq: int
    wip: float
    plan: float

class DowntimeItem(BaseModel):
    model: str
    start_step: int
    end_step: int
    count: int # Number of units of this model unavailable

def load_json(path):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load JSON from {path}: {e}")
        raise

class DataLoader:
    @staticmethod
    def list_scenarios(data_dir):
        if not os.path.exists(data_dir):
            return []
        return [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d)) and d.startswith('scn')]

    def __init__(self, data_dir: str = None, scenario: str = None, raw_data: Dict[str, Any] = None):
        self.data_dir = data_dir
        self.scenario = scenario
        self.raw_data = raw_data
        
        if raw_data:
            self.scenario_path = None
        elif scenario and data_dir:
            self.scenario_path = os.path.join(data_dir, scenario)
        else:
            self.scenario_path = data_dir
            
        self._load_and_validate()

    def _load_and_validate(self):
        try:
            if self.raw_data:
                self.capabilities = [Capability(**c) for c in self.raw_data['capabilities']]
                self.changeover_rules = ChangeoverRules(**self.raw_data['changeover'])
                self.inventory = [InventoryItem(**i) for i in self.raw_data['inventory']]
                self.plan_wip = [PlanWIPItem(**p) for p in self.raw_data['plan_wip']]
                self.downtime = [DowntimeItem(**d) for d in self.raw_data.get('downtime', [])]
            else:
                # Load and Validate Capabilities
                cap_data = load_json(os.path.join(self.scenario_path, 'equipment_capability.json'))
                self.capabilities = [Capability(**c) for c in cap_data['capabilities']]
                
                # Load and Validate Changeover Rules
                co_data = load_json(os.path.join(self.scenario_path, 'changeover_rules.json'))
                self.changeover_rules = ChangeoverRules(**co_data)
                
                # Load and Validate Inventory
                inv_data = load_json(os.path.join(self.scenario_path, 'equipment_inventory.json'))
                self.inventory = [InventoryItem(**i) for i in inv_data['inventory']]
                
                # Load and Validate Plan WIP
                pw_data = load_json(os.path.join(self.scenario_path, 'plan_wip.json'))
                self.plan_wip = [PlanWIPItem(**p) for p in pw_data['production']]

                # Load and Validate Downtime (Optional)
                self.downtime = []
                dt_path = os.path.join(self.scenario_path, 'equipment_downtime.json')
                if os.path.exists(dt_path):
                    dt_data = load_json(dt_path)
                    self.downtime = [DowntimeItem(**d) for d in dt_data.get('downtime', [])]
            
            logger.debug(f"Successfully loaded and validated scenario: {self.scenario or 'default'}")
        except ValidationError as e:
            logger.error(f"Validation error in scenario {self.scenario}: {e}")
            raise
        except FileNotFoundError as e:
            logger.error(f"Missing data file in scenario {self.scenario}: {e}")
            raise

    def get_products(self) -> List[str]:
        return sorted(list(set(p.product for p in self.plan_wip)))

    def get_processes(self) -> List[str]:
        proc_data = []
        seen = set()
        for p in sorted(self.plan_wip, key=lambda x: x.oper_seq):
            if p.process not in seen:
                proc_data.append(p.process)
                seen.add(p.process)
        return proc_data

    def get_models(self) -> List[str]:
        models = set()
        for item in self.inventory:
            models.add(item.model)
        for cap in self.capabilities:
            models.add(cap.model)
        return sorted(models)

    def get_st_map(self) -> Dict[Tuple[str, str], float]:
        st_map = {}
        tmp = {}
        for cap in self.capabilities:
            if not cap.feasible:
                continue
            key = (cap.product, cap.process)
            if cap.st <= 0.0:
                continue
            if key not in tmp:
                tmp[key] = cap.st
            else:
                tmp[key] = min(tmp[key], cap.st)
        st_map.update(tmp)
        return st_map

    def get_initial_wip_plan(self) -> Tuple[Dict[Tuple[str, str], float], Dict[Tuple[str, str], float]]:
        wip_map = {}
        plan_map = {}
        for p in self.plan_wip:
            wip_map[(p.product, p.process)] = p.wip
            plan_map[(p.product, p.process)] = p.plan
        return wip_map, plan_map

    def get_changeover_matrix(self) -> Tuple[Dict[Tuple[str, str, str, str], float], float]:
        co_map = {}
        default = self.changeover_rules.default_time
        for rule in self.changeover_rules.rules:
            key = (rule.from_product, rule.from_process, rule.to_product, rule.to_process)
            co_map[key] = rule.time
        return co_map, default

    def get_total_equipment(self) -> int:
        return sum(item.count for item in self.inventory)
