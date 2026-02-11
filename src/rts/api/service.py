"""
Inference execution interface.

This is the single contact point between the API layer and the inference engine.
The queue worker calls `InferenceService.execute()` — nothing else in the API
layer imports from rts.models directly.

To swap the inference engine, change only this file.
"""

import logging
from types import SimpleNamespace
from typing import Dict, Any

from ..config.config_manager import Config
from ..models.inference import run_inference

logger = logging.getLogger(__name__)


class InferenceService:
    """Thin adapter that translates a job-param dict into a run_inference call."""

    def __init__(self, config: Config):
        self._config = config

    def execute(self, params: dict) -> Dict[str, Any]:
        """
        Run inference synchronously and return results.

        Parameters
        ----------
        params : dict
            Must contain at least 'mode'.
            Optional: 'data_dir', 'scenario', 'timekey', 'model_path'.

        Returns
        -------
        dict  {"output_dir": str, "metrics": dict}
        """
        args = SimpleNamespace(
            mode=params.get("mode", "heuristic"),
            data_dir=params.get("data_dir", "./data"),
            scenario=params.get("scenario"),
            timekey=params.get("timekey"),
            model_path=params.get("model_path", "ppo_eqp_allocator"),
            output="inference_results.csv",
        )

        result = run_inference(args, config=self._config)

        # Normalise return value so the queue layer never touches internals
        return {
            "output_dir": result.get("output_dir") if isinstance(result, dict) else None,
            "metrics": result.get("metrics") if isinstance(result, dict) else None,
        }
