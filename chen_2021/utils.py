import torch
import numpy as np
import random


def fix_random_seed():
  # Same seeds as in the original paper
  torch.manual_seed(999)
  np.random.seed(999)
  random.seed(999)
  torch.cuda.manual_seed_all(999)
  torch.backends.cudnn.benchmark = True
  torch.backends.cudnn.deterministic = True
