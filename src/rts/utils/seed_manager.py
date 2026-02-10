import random
import numpy as np
import torch
import logging

logger = logging.getLogger(__name__)

def set_seed(seed: int):
    if seed is None:
        return
    
    logger.info(f"Setting global seed to {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Ensure reproducible algorithms in torch
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
