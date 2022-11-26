import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

class Destripe_Dataset(Dataset):
    def __init__(self, input_file, label_file, transform=None):
        # Read input and output csv
        self.y = pd.read_csv(label_file)
        self.X = pd.read_csv(input_file)
        self.transform = transform
    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        # Check that original image files match for input and label
        assert self.X.iloc[index, 1] == self.y.iloc[index, 1]

        input_data = self.X.iloc[index, 2:].to_numpy()
        mask_pix = np.fromstring(input_data[-1][1:-1], sep=',')
        input = np.concatenate((input_data[:-1], mask_pix)).astype(np.float32)
        
        label_data = self.y.iloc[index, 2:].to_numpy()
        mask_pix = np.fromstring(label_data[-1][1:-1], sep=',')
        label = np.concatenate((label_data[:-1], mask_pix)).astype(np.float32)

        if self.transform is not None:
            input = self.transform(input)
            label = self.transform(label)

        return input, label

class Photon_Dataset(Dataset):
    def __init__(self,):
        pass
    def __len__(self):
        pass
    def __getitem__(self, index):
        pass