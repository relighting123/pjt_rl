"""
Application composition root.

This is the only file that knows about ALL pieces.
It wires:  Config → Service → JobManager → Routes → FastAPI app
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from ..config.config_manager import Config, load_config
from ..utils.logging_config import setup_logging
from .service import InferenceService
from .queue import JobManager
from .routes import create_router

logger = logging.getLogger(__name__)


def create_app(config: Config = None) -> FastAPI:
    """
    Application factory.

    1. Build InferenceService  (knows how to run inference)
    2. Build JobManager        (queue + worker + scheduler, calls service)
    3. Build Router            (HTTP endpoints, calls manager)
    4. Assemble FastAPI app
    """
    if config is None:
        config = load_config("config.yaml")

    # -- 1. Service layer --------------------------------------------------
    service = InferenceService(config)

    # -- 2. Queue layer ----------------------------------------------------
    schedules = []
    if config.scheduler.enabled:
        schedules = [s.model_dump() for s in config.scheduler.jobs]

    manager = JobManager(
        executor=service.execute,
        queue_size=config.api.queue_size,
        schedules=schedules,
    )

    # -- 3. HTTP layer -----------------------------------------------------
    router = create_router(manager)

    # -- 4. Assemble app ---------------------------------------------------
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
    app.include_router(router)
    app.state.manager = manager  # exposed for tests

    return app


def start_server(config: Config):
    """CLI entry point — called from main.py serve command."""
    import uvicorn
    import matplotlib
    matplotlib.use("Agg")

    setup_logging(log_dir=config.logging.log_dir)
    app = create_app(config)
    uvicorn.run(app, host=config.api.host, port=config.api.port)
