import json
import os

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

class DataLoader:
    @staticmethod
    def list_scenarios(data_dir):
        return [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d)) and d.startswith('scn')]

    def __init__(self, data_dir, scenario=None):
        if scenario:
            scenario_path = os.path.join(data_dir, scenario)
        else:
            # For backward compatibility or if data_dir is already the scenario path
            scenario_path = data_dir
            
        self.capabilities = load_json(os.path.join(scenario_path, 'equipment_capability.json'))['capabilities']
        self.changeover_rules = load_json(os.path.join(scenario_path, 'changeover_rules.json'))
        self.inventory = load_json(os.path.join(scenario_path, 'equipment_inventory.json'))['inventory']
        self.plan_wip = load_json(os.path.join(scenario_path, 'plan_wip.json'))['production']

    def get_products(self):
        return sorted(list(set(p['product'] for p in self.plan_wip)))

    def get_processes(self):
        # Order by oper_seq
        proc_data = []
        seen = set()
        for p in sorted(self.plan_wip, key=lambda x: x['oper_seq']):
            if p['process'] not in seen:
                proc_data.append(p['process'])
                seen.add(p['process'])
        return proc_data

    def get_st_map(self):
        st_map = {}
        for cap in self.capabilities:
            if cap['feasible']:
                st_map[(cap['product'], cap['process'])] = cap['st']
        return st_map

    def get_initial_wip_plan(self):
        wip_map = {}
        plan_map = {}
        for p in self.plan_wip:
            wip_map[(p['product'], p['process'])] = p['wip']
            plan_map[(p['product'], p['process'])] = p['plan']
        return wip_map, plan_map

    def get_changeover_matrix(self):
        # returns (from_prod, from_proc, to_prod, to_proc) -> time
        co_map = {}
        default = self.changeover_rules['default_time']
        for rule in self.changeover_rules['rules']:
            key = (rule['from_product'], rule['from_process'], rule['to_product'], rule['to_process'])
            co_map[key] = rule['time']
        return co_map, default

    def get_total_equipment(self):
        return sum(item['count'] for item in self.inventory)
