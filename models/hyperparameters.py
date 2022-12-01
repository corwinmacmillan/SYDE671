import numpy as np
import torch

D_LEARNING_RATE = 1e-4
D_NUM_EPOCH = 10
D_BATCH_SIZE = 128

P_LEARNING_RATE = 1e-4
P_NUM_EPOCH = 10
P_BATCH_SIZE = 10

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SUMMED_DATA = True