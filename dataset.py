import pandas as pd
import numpy as np
from torch.utils.data import Dataset

class Destripe_Dataset(Dataset):
    def __init__(self, input_file, label_file):
        # Read input and output csv
        self.y = pd.read_csv(label_file)
        self.X = pd.read_csv(input_file)
    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        # Check that original image files match for input and label
        assert self.X.iloc[index, 0] == self.y.iloc[index, 0]

        input = self.X.iloc[index, 1:] # read row and ignore filename
        label = self.y.iloc[index, 1:]

        return input, label

class Photon_Dataset(Dataset):
    def __init__(self,):
        pass
    def __len__(self):
        pass
    def __getitem__(self, index):
        pass