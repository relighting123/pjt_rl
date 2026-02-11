import pytest
import asyncio
import json
import os

from httpx import AsyncClient, ASGITransport
from rts.api.server import create_app, JobManager
from rts.config.config_manager import Config


@pytest.fixture
def app_and_data(tmp_path):
    """Create a minimal data dir and a configured FastAPI app."""
    d = tmp_path / "data"
    d.mkdir()
    scn = d / "scn#1"
    scn.mkdir()

    with open(scn / "equipment_capability.json", "w") as f:
        json.dump({"capabilities": [
            {"product": "P1", "process": "S1", "model": "M1", "st": 10.0, "feasible": True, "initial_count": 1},
            {"product": "P1", "process": "S2", "model": "M1", "st": 20.0, "feasible": True},
        ]}, f)
    with open(scn / "changeover_rules.json", "w") as f:
        json.dump({"default_time": 60.0, "rules": []}, f)
    with open(scn / "equipment_inventory.json", "w") as f:
        json.dump({"inventory": [{"model": "M1", "count": 2}]}, f)
    with open(scn / "plan_wip.json", "w") as f:
        json.dump({"production": [
            {"product": "P1", "process": "S1", "oper_seq": 1, "wip": 100, "plan": 50},
            {"product": "P1", "process": "S2", "oper_seq": 2, "wip": 0, "plan": 50},
        ]}, f)

    cfg = Config()
    cfg.db.enabled = False
    app = create_app(cfg)
    return app, str(d)


@pytest.mark.asyncio
async def test_health(app_and_data):
    app, data_dir = app_and_data
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert "queue_size" in body


@pytest.mark.asyncio
async def test_submit_and_complete_job(app_and_data):
    app, data_dir = app_and_data
    mgr: JobManager = app.state.manager

    # Start the worker manually (lifespan doesn't run in test transport)
    mgr.start()
    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            # Submit
            resp = await client.post("/infer", json={
                "mode": "heuristic",
                "scenario": "scn#1",
                "data_dir": data_dir,
            })
            assert resp.status_code == 200
            body = resp.json()
            assert body["status"] == "queued"
            job_id = body["job_id"]

            # Wait for worker to finish
            for _ in range(20):
                await asyncio.sleep(0.5)
                resp2 = await client.get(f"/jobs/{job_id}")
                if resp2.json()["status"] in ("completed", "failed"):
                    break

            result = resp2.json()
            assert result["status"] == "completed", f"Job failed: {result.get('error')}"
            assert result["metrics"] is not None
            assert "achievement_rate" in result["metrics"]
            assert result["result_path"] is not None
    finally:
        mgr.stop()


@pytest.mark.asyncio
async def test_queue_status(app_and_data):
    app, data_dir = app_and_data
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/queue")
        assert resp.status_code == 200
        body = resp.json()
        for key in ("pending", "running", "completed", "failed", "queue_size"):
            assert key in body


@pytest.mark.asyncio
async def test_list_jobs(app_and_data):
    app, data_dir = app_and_data
    mgr: JobManager = app.state.manager
    mgr.start()
    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            await client.post("/infer", json={
                "mode": "heuristic", "scenario": "scn#1", "data_dir": data_dir,
            })
            await client.post("/infer", json={
                "mode": "heuristic", "scenario": "scn#1", "data_dir": data_dir,
            })

            resp = await client.get("/jobs")
            assert resp.status_code == 200
            assert len(resp.json()) == 2
    finally:
        mgr.stop()


@pytest.mark.asyncio
async def test_job_not_found(app_and_data):
    app, data_dir = app_and_data
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/jobs/nonexistent")
        assert resp.status_code == 404
