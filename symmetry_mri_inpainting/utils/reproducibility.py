import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """
    Set the seed for reproducibility.
    This function sets the seed for the random, numpy, and torch libraries.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
