import yaml
import os
import logging
from pydantic import BaseModel, Field
from typing import Dict, Any

logger = logging.getLogger(__name__)

class RewardConfig(BaseModel):
    last_step_weight: float = 50.0
    total_production_weight: float = 20.0
    changeover_penalty: float = 5.0
    final_achievement_bonus: float = 100.0

class EnvConfig(BaseModel):
    max_steps: int = 24
    reward: RewardConfig = RewardConfig()

class TrainConfig(BaseModel):
    timesteps: int = 60000
    learning_rate: float = 0.0003
    batch_size: int = 128
    n_steps: int = 1024
    ent_coef: float = 0.01
    seed: int = 42

class LoggingConfig(BaseModel):
    level: str = "INFO"
    log_dir: str = "logs"

class Config(BaseModel):
    env: EnvConfig = EnvConfig()
    train: TrainConfig = TrainConfig()
    logging: LoggingConfig = LoggingConfig()

def load_config(path: str = "config.yaml") -> Config:
    if not os.path.exists(path):
        logger.warning(f"Config file {path} not found. Using defaults.")
        return Config()
    
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
            return Config(**data)
    except Exception as e:
        logger.error(f"Failed to load config: {e}. Using defaults.")
        return Config()
