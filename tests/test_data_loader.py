import pytest
import os
import shutil
import json
from rts.data.data_loader import DataLoader

@pytest.fixture
def mock_data_dir(tmp_path):
    d = tmp_path / "data"
    d.mkdir()
    scn = d / "scn#1"
    scn.mkdir()
    
    # Mock files
    with open(scn / "equipment_capability.json", "w") as f:
        json.dump({"capabilities": [{"product": "P1", "process": "S1", "model": "M1", "st": 10.0, "feasible": True}]}, f)
    with open(scn / "changeover_rules.json", "w") as f:
        json.dump({"default_time": 60.0, "rules": []}, f)
    with open(scn / "equipment_inventory.json", "w") as f:
        json.dump({"inventory": [{"model": "M1", "count": 5}]}, f)
    with open(scn / "plan_wip.json", "w") as f:
        json.dump({"production": [{"product": "P1", "process": "S1", "oper_seq": 1, "wip": 100, "plan": 50}]}, f)
    
    return str(d)

def test_data_loader_init(mock_data_dir):
    loader = DataLoader(mock_data_dir, "scn#1")
    assert loader.get_products() == ["P1"]
    assert loader.get_processes() == ["S1"]
    assert loader.get_total_equipment() == 5

def test_data_loader_list_scenarios(mock_data_dir):
    scenarios = DataLoader.list_scenarios(mock_data_dir)
    assert "scn#1" in scenarios
