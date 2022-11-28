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
    ToTensor,
    Compose,
)

from dataset import Destripe_Dataset, Photon_Dataset

def split_destripe(
    destripe_data_csv,
    destripe_data_path,
    split_size = 0.2
):
    '''
    :params:
        destripe_data_csv: .csv file of destripe inputs and labels from noisy_img.py -> generate_destripe_data()
        destripe_data_path: path to parent folder of destripe train and val folders
        split_size: validation dataset size for train_test_split()
    '''
        
    destripe_path_train = os.path.join(destripe_data_path, 'train')
    destripe_path_val = os.path.join(destripe_data_path, 'val')

    # Check for train/val folders
    if not os.path.exists(destripe_path_train):
        os.makedirs(destripe_path_train)
    if not os.path.exists(destripe_path_val):
        os.makedirs(destripe_path_val)

    # Read destripe data
    df = pd.read_csv(destripe_data_csv)

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
        label_file=label_train_files,
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle)

    val_ds = Destripe_Dataset(
        input_file=input_val_files,
        label_file=label_val_files,
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=shuffle)

    return train_loader, val_loader

def photon_loaders(
    input_train_path,
    label_train_path,
    input_val_path,
    label_val_path,
    batch_size,
    shuffle=False,
):
    '''
    :params:
        input_train_path: model input train path
        label_train_path: model label train path
        input_val_path: model input val path
        label_val_path: model label val path
        batch_size: batch size for dataloader
        shuffle=False: bool to shuffle loader
    '''
    train_ds = Photon_Dataset(
        image_dir=input_train_path,
        label_dir=label_train_path,
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle)

    val_ds = Photon_Dataset(
        image_dir=input_val_path,
        label_dir=label_val_path,
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=shuffle)

    return train_loader, val_loader

def L1_loss(prediction, label):
    L1_abs_loss = nn.L1Loss()
    value = L1_abs_loss(prediction, label).item()
    return value
