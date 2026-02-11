"""
Queue infrastructure.

JobManager owns: job store, asyncio.Queue, single worker, scheduler timers.
It does NOT know about FastAPI, inference, or project config —
it receives an `executor` callable from the outside.

    executor(params: dict) -> dict   # synchronous, called in thread pool
"""

import asyncio
import uuid
import logging
from datetime import datetime
from typing import Callable, Optional, Dict, List

logger = logging.getLogger(__name__)


class JobManager:
    """
    Single-worker job queue with optional timer-based scheduling.

    Parameters
    ----------
    executor : callable(dict) -> dict
        Synchronous function that processes one job.
        Receives the job's parameter dict, returns {"output_dir", "metrics"}.
    queue_size : int
        Maximum number of jobs that can wait in the queue.
    schedules : list[dict], optional
        Timer definitions.  Each dict may contain:
          name, mode, timekey, scenario, data_dir, model_path, interval_seconds
    """

    def __init__(
        self,
        executor: Callable[[dict], dict],
        queue_size: int = 100,
        schedules: Optional[List[dict]] = None,
    ):
        self._executor = executor
        self.queue: asyncio.Queue = asyncio.Queue(maxsize=queue_size)
        self.jobs: Dict[str, dict] = {}
        self._schedules = schedules or []
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
        """Put a job into the queue.  Raises RuntimeError if full."""
        if self.queue.full():
            raise RuntimeError("Queue is full")
        await self.queue.put(job)

    def get_job(self, job_id: str) -> Optional[dict]:
        return self.jobs.get(job_id)

    def list_jobs(self) -> List[dict]:
        return sorted(
            self.jobs.values(),
            key=lambda j: j["created_at"],
            reverse=True,
        )

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

    async def _process_job(self, job: dict):
        job_id = job["job_id"]
        job["status"] = "running"
        job["started_at"] = datetime.now().isoformat()
        logger.info(
            f"[Worker] Processing {job_id}  mode={job.get('mode')}  "
            f"timekey={job.get('timekey')}  scenario={job.get('scenario')}"
        )
        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, lambda: self._executor(job)
            )
            job["status"] = "completed"
            job["completed_at"] = datetime.now().isoformat()
            job["result_path"] = result.get("output_dir")
            job["metrics"] = result.get("metrics")
            logger.info(f"[Worker] Job {job_id} completed")
        except Exception as e:
            job["status"] = "failed"
            job["completed_at"] = datetime.now().isoformat()
            job["error"] = str(e)
            logger.error(f"[Worker] Job {job_id} failed: {e}")

    async def _worker_loop(self):
        logger.info("[Worker] Started — processing jobs sequentially")
        while True:
            job = await self.queue.get()
            try:
                await self._process_job(job)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[Worker] Unexpected error: {e}")
            finally:
                self.queue.task_done()

    # -- scheduler ---------------------------------------------------------

    async def _scheduler_loop(self, sched: dict):
        name = sched.get("name", "unnamed")
        interval = sched.get("interval_seconds", 3600)
        logger.info(f"[Scheduler] '{name}' started  interval={interval}s")

        while True:
            await asyncio.sleep(interval)

            timekey = sched.get("timekey")
            if timekey == "auto":
                timekey = datetime.now().strftime("%Y%m%d%H%M%S")

            job = self.create_job(
                source="scheduler",
                mode=sched.get("mode", "heuristic"),
                timekey=timekey,
                scenario=sched.get("scenario"),
                data_dir=sched.get("data_dir", "./data"),
                model_path=sched.get("model_path", "ppo_eqp_allocator"),
            )
            try:
                await self.submit(job)
                logger.info(f"[Scheduler] '{name}' submitted job {job['job_id']}")
            except RuntimeError:
                job["status"] = "failed"
                job["error"] = "Queue full"
                logger.warning(f"[Scheduler] '{name}' could not submit — queue full")

    # -- lifecycle ---------------------------------------------------------

    def start(self):
        """Start background worker and schedulers.  Call inside a running loop."""
        self._worker_task = asyncio.create_task(self._worker_loop())
        for sched in self._schedules:
            task = asyncio.create_task(self._scheduler_loop(sched))
            self._scheduler_tasks.append(task)

    def stop(self):
        """Cancel background tasks."""
        if self._worker_task:
            self._worker_task.cancel()
        for t in self._scheduler_tasks:
            t.cancel()
