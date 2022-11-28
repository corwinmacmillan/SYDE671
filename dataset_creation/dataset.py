import pandas as pd
import numpy as np
import torch
import re
import os
import cv2
from torch.utils.data import Dataset
from utils.planetaryimageEDR import PDS3ImageEDR

class Destripe_Dataset(Dataset):
    '''
    DestripeNet dataset
    :params:
        self.y: label .csv file
        self.X: input .csv file
        self.transform: transforms
    '''
    def __init__(self, input_file, label_file, image_path, transform=None):
        # Read input and output csv
        self.y = pd.read_csv(label_file)
        self.X = pd.read_csv(input_file)
        self.image_path = image_path
        self.transform = transform
    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        # Check that original image files match for input and label
        assert self.X.iloc[index, 1] == self.y.iloc[index, 1]

        input_data = self.X.iloc[index, 2:].to_numpy()
        mask_pix = np.fromstring(input_data[-1][1:-1], sep=',')
        input = np.concatenate((input_data[:-1], mask_pix)).astype(np.float32)
        input = input.reshape(38, 1)
        
        # label_data = self.y.iloc[index, 2:].to_numpy()[0]
        # label_data = re.sub('\n', '', label_data)
        # label = np.fromstring(label_data[1:-1], sep=' ').astype(np.float32)

        line_index = self.y.iloc[index, 2]
        filename = self.y.iloc[index, 1]
        I = PDS3ImageEDR.open(os.path.join(self.image_path, filename))
        label = I.image[line_index, :].astype(np.float32)

        if self.transform is not None:
            input = self.transform(input)
            label = self.transform(label)

        return input, label

class Photon_Dataset(Dataset):
    '''
    PhotonNet dataset
    :params:
        self.image_dir: path to input images directory
        self.label_dir: path to label images directory
        self.transform: transforms
    '''
    def __init__(self, image_dir, label_dir, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform

        self.images = os.listdir(image_dir)
        self.labels = os.listdir(label_dir)

    def __len__(self):
        return len(self.images)
    def __getitem__(self, index):
        image_path = os.path.join(self.image_dir, self.images[index])
        label_path = os.path.join(self.label_dir, self.labels[index])

        image = cv2.imread(image_path)
        label = cv2.imread(label_path)

        '''
        INCOMPLETE
        1) need to apply destripe output, inverse non-linearity, and flatfield to input image
        2) need to apply J-S to obtain training label
        '''

        if self.transform is not None:
            image = self.transform(image)
            label = self.transform(label)

        return image, label