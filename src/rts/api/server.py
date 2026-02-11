"""
RTS Inference API Server

Queue-based inference service.
- Event-driven: POST /infer pushes a job to the queue
- Timer-driven: scheduler periodically pushes jobs based on config.yaml
- Single worker processes jobs sequentially (one at a time)
"""

import asyncio
import uuid
import logging
from datetime import datetime
from types import SimpleNamespace
from typing import Optional, Dict, Any, List
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from ..config.config_manager import Config, load_config
from ..utils.logging_config import setup_logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Request / Response Models
# ---------------------------------------------------------------------------

class InferenceRequest(BaseModel):
    mode: str = Field(default="heuristic", description="rl or heuristic")
    timekey: Optional[str] = Field(default=None, description="DB RULE_TIMEKEY (DB mode)")
    scenario: Optional[str] = Field(default=None, description="Scenario directory name (file mode)")
    data_dir: str = Field(default="./data")
    model_path: str = Field(default="ppo_eqp_allocator")


class JobResponse(BaseModel):
    job_id: str
    status: str
    mode: str
    timekey: Optional[str] = None
    scenario: Optional[str] = None
    source: str = "api"
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    result_path: Optional[str] = None
    metrics: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class QueueStatus(BaseModel):
    pending: int
    running: int
    completed: int
    failed: int
    queue_size: int


# ---------------------------------------------------------------------------
# Job Manager  (queue + worker + scheduler in one class)
# ---------------------------------------------------------------------------

class JobManager:
    """Single-worker queue with optional timer-based scheduling."""

    def __init__(self, config: Config):
        self.config = config
        self.queue: asyncio.Queue = asyncio.Queue(maxsize=config.api.queue_size)
        self.jobs: Dict[str, dict] = {}
        self._worker_task: Optional[asyncio.Task] = None
        self._scheduler_tasks: List[asyncio.Task] = []

    # -- job CRUD ----------------------------------------------------------

    def create_job(self, source: str = "api", **kwargs) -> dict:
        job_id = uuid.uuid4().hex[:8]
        job = {
            "job_id": job_id,
            "status": "queued",
            "source": source,
            "created_at": datetime.now().isoformat(),
            "started_at": None,
            "completed_at": None,
            "result_path": None,
            "metrics": None,
            "error": None,
            **kwargs,
        }
        self.jobs[job_id] = job
        return job

    async def submit(self, job: dict):
        if self.queue.full():
            raise RuntimeError("Queue is full")
        await self.queue.put(job)

    def status_summary(self) -> dict:
        statuses = [j["status"] for j in self.jobs.values()]
        return {
            "pending": statuses.count("queued"),
            "running": statuses.count("running"),
            "completed": statuses.count("completed"),
            "failed": statuses.count("failed"),
            "queue_size": self.queue.qsize(),
        }

    # -- worker ------------------------------------------------------------

    async def _run_inference_job(self, job: dict):
        """Execute a single inference job (runs in thread pool)."""
        from ..models.inference import run_inference

        args = SimpleNamespace(
            mode=job["mode"],
            data_dir=job.get("data_dir", "./data"),
            scenario=job.get("scenario"),
            timekey=job.get("timekey"),
            model_path=job.get("model_path", "ppo_eqp_allocator"),
            output="inference_results.csv",
        )

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, lambda: run_inference(args, config=self.config)
        )
        return result

    async def _worker_loop(self):
        logger.info("[Worker] Started - processing jobs sequentially")
        while True:
            job = await self.queue.get()
            job_id = job["job_id"]
            job["status"] = "running"
            job["started_at"] = datetime.now().isoformat()
            logger.info(
                f"[Worker] Processing {job_id}  mode={job['mode']}  "
                f"timekey={job.get('timekey')}  scenario={job.get('scenario')}"
            )
            try:
                result = await self._run_inference_job(job)
                job["status"] = "completed"
                job["completed_at"] = datetime.now().isoformat()
                if isinstance(result, dict):
                    job["result_path"] = result.get("output_dir")
                    job["metrics"] = result.get("metrics")
                logger.info(f"[Worker] Job {job_id} completed")
            except Exception as e:
                job["status"] = "failed"
                job["completed_at"] = datetime.now().isoformat()
                job["error"] = str(e)
                logger.error(f"[Worker] Job {job_id} failed: {e}")
            finally:
                self.queue.task_done()

    # -- scheduler ---------------------------------------------------------

    async def _scheduler_loop(self, sched_cfg: dict):
        name = sched_cfg.get("name", "unnamed")
        interval = sched_cfg.get("interval_seconds", 3600)
        logger.info(f"[Scheduler] '{name}' started  interval={interval}s")

        while True:
            await asyncio.sleep(interval)

            timekey = sched_cfg.get("timekey")
            if timekey == "auto":
                timekey = datetime.now().strftime("%Y%m%d%H%M%S")

            job = self.create_job(
                source="scheduler",
                mode=sched_cfg.get("mode", "heuristic"),
                timekey=timekey,
                scenario=sched_cfg.get("scenario"),
                data_dir=sched_cfg.get("data_dir", "./data"),
                model_path=sched_cfg.get("model_path", "ppo_eqp_allocator"),
            )
            try:
                await self.submit(job)
                logger.info(f"[Scheduler] '{name}' submitted job {job['job_id']}")
            except RuntimeError:
                job["status"] = "failed"
                job["error"] = "Queue full"
                logger.warning(f"[Scheduler] '{name}' could not submit - queue full")

    # -- lifecycle ---------------------------------------------------------

    def start(self):
        self._worker_task = asyncio.create_task(self._worker_loop())

        if self.config.scheduler.enabled:
            for sched in self.config.scheduler.jobs:
                task = asyncio.create_task(
                    self._scheduler_loop(sched.model_dump())
                )
                self._scheduler_tasks.append(task)

    def stop(self):
        if self._worker_task:
            self._worker_task.cancel()
        for t in self._scheduler_tasks:
            t.cancel()


# ---------------------------------------------------------------------------
# FastAPI Application Factory
# ---------------------------------------------------------------------------

def create_app(config: Config = None) -> FastAPI:
    if config is None:
        config = load_config("config.yaml")

    manager = JobManager(config)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        manager.start()
        logger.info(
            f"RTS API ready  host={config.api.host}  port={config.api.port}  "
            f"scheduler={'ON' if config.scheduler.enabled else 'OFF'}"
        )
        yield
        manager.stop()
        logger.info("RTS API stopped")

    app = FastAPI(
        title="RTS Inference API",
        description="Queue-based inference service for EQP Allocation Optimizer",
        lifespan=lifespan,
    )

    # Store on app.state so routes (and tests) can always access it
    app.state.manager = manager

    # -- routes ------------------------------------------------------------

    @app.post("/infer", response_model=JobResponse)
    async def submit_inference(req: InferenceRequest):
        """Submit an inference job to the queue."""
        mgr: JobManager = app.state.manager
        job = mgr.create_job(
            source="api",
            mode=req.mode,
            timekey=req.timekey,
            scenario=req.scenario,
            data_dir=req.data_dir,
            model_path=req.model_path,
        )
        try:
            await mgr.submit(job)
        except RuntimeError as e:
            raise HTTPException(status_code=503, detail=str(e))
        return job

    @app.get("/jobs", response_model=List[JobResponse])
    async def list_jobs():
        """List all jobs (recent first)."""
        mgr: JobManager = app.state.manager
        return sorted(
            mgr.jobs.values(),
            key=lambda j: j["created_at"],
            reverse=True,
        )

    @app.get("/jobs/{job_id}", response_model=JobResponse)
    async def get_job(job_id: str):
        """Get status and result of a specific job."""
        mgr: JobManager = app.state.manager
        job = mgr.jobs.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        return job

    @app.get("/queue", response_model=QueueStatus)
    async def queue_status():
        """Current queue statistics."""
        mgr: JobManager = app.state.manager
        return mgr.status_summary()

    @app.get("/health")
    async def health():
        mgr: JobManager = app.state.manager
        return {"status": "ok", "queue_size": mgr.queue.qsize()}

    return app


# ---------------------------------------------------------------------------
# Entry point (called from main.py serve command)
# ---------------------------------------------------------------------------

def start_server(config: Config):
    import uvicorn
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend for server

    setup_logging(log_dir=config.logging.log_dir)
    app = create_app(config)
    uvicorn.run(app, host=config.api.host, port=config.api.port)
