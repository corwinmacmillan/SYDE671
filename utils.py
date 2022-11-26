import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from torch.utils.data import (
    DataLoader,
    Dataset,    
) 

from torchvision.io import read_image

from torchvision.transforms import (
    RandomHorizontalFlip,
    RandomVerticalFlip,
    RandomRotation,
)

from dataset import Destripe_Dataset

def split_destripe(
    destripe_data,
    destripe_path,
    split_size = 0.2
):
    '''
    :params:
        destripe_data: csv of destripe inputs and labels from noisy_img.py
        destripe_path: path to destripe train and val folders
        split_size: validation dataset size for train_test_split()
    '''
        
    destripe_path_train = os.path.join(destripe_path, 'train')
    destripe_path_val = os.path.join(destripe_path, 'val')

    # Check for train/val folders
    if not os.path.exists(destripe_path_train):
        os.makedirs(destripe_path_train)
    if not os.path.exists(destripe_path_val):
        os.makedirs(destripe_path_val)

    # Read destripe data
    df = pd.read_csv(destripe_data)

    # Split data into inputs and labels
    y = df[['Filename', 'Pixel_line']]
    X = df.drop('Pixel_line', axis=1)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=split_size)

    # Save training and validation data
    X_train.to_csv(os.path.join(destripe_path_train, 'train_inputs.csv'))
    y_train.to_csv(os.path.join(destripe_path_train, 'train_labels.csv'))

    X_val.to_csv(os.path.join(destripe_path_val, 'val_inputs.csv'))
    y_val.to_csv(os.path.join(destripe_path_val, 'val_labels.csv'))
    

def destripe_loaders(
    input_train_files,
    label_train_files,
    input_val_files,
    label_val_files,
    batch_size,
    shuffle=False,
):
    '''
    :params:
        input_train_files: model input train files
        label_train_files: model label train files
        input_val_files: model input val files
        label_val_files: model label val files
        batch_size: batch size for dataloader
        shuffle=False: bool to shuffle loader
    '''
    train_ds = Destripe_Dataset(
        input_file=input_train_files, 
        label_file=label_train_files
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle)

    val_ds = Destripe_Dataset(
        input_file=input_val_files,
        label_file=label_val_files
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=shuffle)

    return train_loader, val_loader

def photon_loaders(
    train_files,
    val_files,
    batch_size,
    shuffle=False,
):
    pass

