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


@pytest.fixture
def multi_scenario_data_dir(tmp_path):
    """Create a data dir with multiple scenarios that have different entities.
    This simulates the real-world case where sync-db creates a scenario alongside
    existing training scenarios with different products/models."""
    d = tmp_path / "data"
    d.mkdir()
    
    # Scenario 1: ProductA with Basic model (similar to DB sync'd data)
    scn1 = d / "scn#1"
    scn1.mkdir()
    with open(scn1 / "equipment_capability.json", "w") as f:
        json.dump({"capabilities": [
            {"product": "ProductA", "process": "Step_10", "model": "Basic", "st": 10.0, "feasible": True, "initial_count": 1},
            {"product": "ProductA", "process": "Step_20", "model": "Basic", "st": 8.0, "feasible": True, "initial_count": 0},
        ]}, f)
    with open(scn1 / "changeover_rules.json", "w") as f:
        json.dump({"default_time": 60.0, "rules": []}, f)
    with open(scn1 / "equipment_inventory.json", "w") as f:
        json.dump({"inventory": [{"model": "Basic", "count": 1}]}, f)
    with open(scn1 / "plan_wip.json", "w") as f:
        json.dump({"production": [
            {"product": "ProductA", "process": "Step_10", "oper_seq": 10, "wip": 40, "plan": 36},
            {"product": "ProductA", "process": "Step_20", "oper_seq": 20, "wip": 0, "plan": 36}
        ]}, f)
    
    # Scenario 2: Different products and models (like training data)
    scn2 = d / "scn#2"
    scn2.mkdir()
    with open(scn2 / "equipment_capability.json", "w") as f:
        json.dump({"capabilities": [
            {"product": "Product_X", "process": "Step_10", "model": "TypeA", "st": 10.0, "feasible": True, "initial_count": 1},
            {"product": "Product_X", "process": "Step_20", "model": "TypeB", "st": 15.0, "feasible": True, "initial_count": 1},
            {"product": "Product_Y", "process": "Step_10", "model": "TypeA", "st": 12.0, "feasible": True, "initial_count": 1},
        ]}, f)
    with open(scn2 / "changeover_rules.json", "w") as f:
        json.dump({"default_time": 120.0, "rules": []}, f)
    with open(scn2 / "equipment_inventory.json", "w") as f:
        json.dump({"inventory": [{"model": "TypeA", "count": 2}, {"model": "TypeB", "count": 1}]}, f)
    with open(scn2 / "plan_wip.json", "w") as f:
        json.dump({"production": [
            {"product": "Product_X", "process": "Step_10", "oper_seq": 10, "wip": 200, "plan": 150},
            {"product": "Product_X", "process": "Step_20", "oper_seq": 20, "wip": 50, "plan": 150},
            {"product": "Product_Y", "process": "Step_10", "oper_seq": 10, "wip": 100, "plan": 80},
        ]}, f)
    
    return str(d)


def test_db_vs_json_consistency(multi_scenario_data_dir):
    """Test that DB-based (raw_data) and file-based (fixed_scenario) inference
    produce identical entity mappings, scales, observations, and results.
    
    This is the core regression test for the DB vs JSON inference difference bug."""
    
    # Simulate raw_data as it would come from DBManager.fetch_data()
    raw_data = {
        "capabilities": [
            {"product": "ProductA", "process": "Step_10", "model": "Basic", "st": 10.0, "feasible": True, "initial_count": 1},
            {"product": "ProductA", "process": "Step_20", "model": "Basic", "st": 8.0, "feasible": True, "initial_count": 0},
        ],
        "changeover": {"default_time": 60.0, "rules": []},
        "inventory": [{"model": "Basic", "count": 1}],
        "plan_wip": [
            {"product": "ProductA", "process": "Step_10", "oper_seq": 10, "wip": 40, "plan": 36},
            {"product": "ProductA", "process": "Step_20", "oper_seq": 20, "wip": 0, "plan": 36},
        ],
        "downtime": []
    }
    
    # Path 1: DB-direct mode (raw_data)
    env_db = ProductionEnv(multi_scenario_data_dir, fixed_scenario="scn#1", raw_data=raw_data)
    obs_db, _ = env_db.reset()
    
    # Path 2: File-based mode (fixed_scenario) - data dir has MULTIPLE scenarios
    env_file = ProductionEnv(multi_scenario_data_dir, fixed_scenario="scn#1")
    obs_file, _ = env_file.reset()
    
    # 1. Entity lists must be identical
    assert env_db.products == env_file.products, \
        f"Products differ: DB={env_db.products} vs File={env_file.products}"
    assert env_db.processes == env_file.processes, \
        f"Processes differ: DB={env_db.processes} vs File={env_file.processes}"
    assert env_db.models == env_file.models, \
        f"Models differ: DB={env_db.models} vs File={env_file.models}"
    
    # 2. Index mappings must be identical
    assert env_db.prod_idx == env_file.prod_idx, \
        f"prod_idx differ: DB={env_db.prod_idx} vs File={env_file.prod_idx}"
    assert env_db.proc_idx == env_file.proc_idx
    assert env_db.model_idx == env_file.model_idx
    
    # 3. Global scales must be identical
    assert env_db.max_wip_global == env_file.max_wip_global, \
        f"max_wip differs: DB={env_db.max_wip_global} vs File={env_file.max_wip_global}"
    assert env_db.max_plan_global == env_file.max_plan_global, \
        f"max_plan differs: DB={env_db.max_plan_global} vs File={env_file.max_plan_global}"
    assert env_db.max_eqp_global == env_file.max_eqp_global, \
        f"max_eqp differs: DB={env_db.max_eqp_global} vs File={env_file.max_eqp_global}"
    
    # 4. Initial state matrices must be identical
    np.testing.assert_array_equal(env_db.wip, env_file.wip)
    np.testing.assert_array_equal(env_db.plan, env_file.plan)
    np.testing.assert_array_equal(env_db.active_eqp, env_file.active_eqp)
    np.testing.assert_array_equal(env_db.st_matrix, env_file.st_matrix)
    
    # 5. Observations must be identical
    np.testing.assert_array_almost_equal(obs_db, obs_file, decimal=6,
        err_msg="Initial observations differ between DB and file paths")
    
    # 6. Run a few steps with the same actions and verify identical outputs
    actions = [0, 1, 0, 2, 0]
    for action in actions:
        obs_db, rew_db, term_db, _, _ = env_db.step(action)
        obs_file, rew_file, term_file, _, _ = env_file.step(action)
        
        np.testing.assert_array_almost_equal(obs_db, obs_file, decimal=6,
            err_msg=f"Observations differ at action {action}")
        assert abs(rew_db - rew_file) < 1e-6, \
            f"Rewards differ at action {action}: DB={rew_db} vs File={rew_file}"
        assert term_db == term_file


def test_fixed_scenario_entity_isolation(multi_scenario_data_dir):
    """Test that fixed_scenario only discovers entities from the target scenario,
    NOT from all scenarios in the data directory."""
    
    # With fixed_scenario="scn#1", should only see ProductA and Basic
    env = ProductionEnv(multi_scenario_data_dir, fixed_scenario="scn#1")
    
    # Should NOT include Product_X, Product_Y, TypeA, TypeB from scn#2
    assert "Product_X" not in env.products
    assert "Product_Y" not in env.products
    assert "TypeA" not in env.models
    assert "TypeB" not in env.models
    
    # Should include entities from scn#1 only
    assert "ProductA" in env.products
    assert "Basic" in env.models
    assert env.real_num_prods == 1  # Only ProductA
    assert len([m for m in env.models if not m.startswith("PAD")]) == 1  # Only Basic


def test_training_mode_scans_all_scenarios(multi_scenario_data_dir):
    """Test that training mode (no fixed_scenario) still discovers ALL entities."""
    
    env = ProductionEnv(multi_scenario_data_dir)
    
    # Should include entities from ALL scenarios
    assert "ProductA" in env.products
    assert "Product_X" in env.products
    assert "Product_Y" in env.products
    assert "Basic" in env.models
    assert "TypeA" in env.models
    assert "TypeB" in env.models
