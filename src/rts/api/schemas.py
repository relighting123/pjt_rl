"""
API data contracts.

Pure Pydantic models — no business logic, no framework imports.
"""

from typing import Optional, Dict, Any
from pydantic import BaseModel, Field


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
