import numpy as np
import torch

LEARNING_RATE = 1e-4
NUM_EPOCH = 10
BATCH_SIZE = 1
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SUMMED_DATA = True