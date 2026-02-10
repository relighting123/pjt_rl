import pytest
import numpy as np
from rts.env.factory_env import ProductionEnv
from rts.data.data_loader import DataLoader
import json

@pytest.fixture
def mock_data_dir(tmp_path):
    d = tmp_path / "data"
    d.mkdir()
    scn = d / "scn#1"
    scn.mkdir()
    
    # Simple setup
    with open(scn / "equipment_capability.json", "w") as f:
        json.dump({"capabilities": [
            {"product": "P1", "process": "S1", "model": "M1", "st": 10.0, "feasible": True, "initial_count": 1},
            {"product": "P1", "process": "S2", "model": "M1", "st": 20.0, "feasible": True}
        ]}, f)
    with open(scn / "changeover_rules.json", "w") as f:
        json.dump({"default_time": 60.0, "rules": []}, f)
    with open(scn / "equipment_inventory.json", "w") as f:
        json.dump({"inventory": [{"model": "M1", "count": 2}]}, f)
    with open(scn / "plan_wip.json", "w") as f:
        json.dump({"production": [
            {"product": "P1", "process": "S1", "oper_seq": 1, "wip": 100, "plan": 50},
            {"product": "P1", "process": "S2", "oper_seq": 2, "wip": 0, "plan": 50}
        ]}, f)
    
    return str(d)

def test_env_reset(mock_data_dir):
    env = ProductionEnv(mock_data_dir, fixed_scenario="scn#1")
    obs, info = env.reset()
    assert obs.shape[0] == env.observation_space.shape[0]
    assert env.active_eqp.sum() == 1

def test_env_step(mock_data_dir):
    env = ProductionEnv(mock_data_dir, fixed_scenario="scn#1")
    env.reset()
    # Action 0: Stay
    obs, reward, terminated, truncated, info = env.step(0)
    assert not terminated
    assert env.current_step == 1
    # Production should happen at S1
    assert env.produced[0, 0] > 0
    assert env.wip[0, 1] > 0 # Flow to S2
