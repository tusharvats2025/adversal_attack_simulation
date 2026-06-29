import random
import numpy as np
import torch
import os

def set_seed(seed: int = 42, cpu_threads: int = 4):
    """
    Set random seeds for reproducibility acros all libraries.

    Args:
      seed: Integer seed value (default: 42)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # CPU-specific optimisation

    torch.set_num_threads(cpu_threads)
    os.environ["OMP_NUM_THREADS"] = str(cpu_threads)
    os.environ["MKL_NUM_THREADS"] = str(cpu_threads)
    
    
    # Only set CUDA seeds if CUDA is available 
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        print(f"[Seed] Set random seed to {seed} (CUDA available, {cpu_threads})")
    else:
        print(f"[Seed] Set random seed to {seed} (CPU mode, {cpu_threads} threads)")

