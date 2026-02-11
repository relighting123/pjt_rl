"""
HTTP route definitions.

Knows only about schemas and the JobManager interface.
Does NOT import inference, Config, or service.
"""

from typing import List

from fastapi import APIRouter, HTTPException

from .schemas import InferenceRequest, JobResponse, QueueStatus
from .queue import JobManager


def create_router(manager: JobManager) -> APIRouter:
    """Build an APIRouter wired to the given JobManager."""
    router = APIRouter()

    @router.post("/infer", response_model=JobResponse)
    async def submit_inference(req: InferenceRequest):
        """Submit an inference job to the queue."""
        job = manager.create_job(
            source="api",
            mode=req.mode,
            timekey=req.timekey,
            scenario=req.scenario,
            data_dir=req.data_dir,
            model_path=req.model_path,
        )
        try:
            await manager.submit(job)
        except RuntimeError as e:
            raise HTTPException(status_code=503, detail=str(e))
        return job

    @router.get("/jobs", response_model=List[JobResponse])
    async def list_jobs():
        """List all jobs (recent first)."""
        return manager.list_jobs()

    @router.get("/jobs/{job_id}", response_model=JobResponse)
    async def get_job(job_id: str):
        """Get status and result of a specific job."""
        job = manager.get_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        return job

    @router.get("/queue", response_model=QueueStatus)
    async def queue_status():
        """Current queue statistics."""
        return manager.status_summary()

    @router.get("/health")
    async def health():
        return {"status": "ok", "queue_size": manager.queue.qsize()}

    return router
