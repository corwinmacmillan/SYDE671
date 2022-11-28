import os
import torch.optim as optim
from models import hyperparameters as hp
import torch.nn as nn

from utils.util import (
    split_destripe,
    destripe_loaders,
)

from models.train import (
    destripe_train_fn,
)

from models.tensorboard_utils import (
    inspect_model,
)

from models.destripe import DestripeNet

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('destripe/tensorboard')

# Conditional functions in main()
SPLIT_DESTRIPE = True
'''
SPLIT_DESTRIPE: split the data .csv file generated in noisy_img.py -> 
                generate_destripe_data() into training and validation folders
'''

# Paths
DESTRIPE_DATA_CSV = '/media/panlab/EXTERNALHDD/dark_summed/dark_summed_data.csv'
DESTRIPE_DATA_PATH = '/media/panlab/CHARVIHDD/SYDE671/DestripeNet'
MODEL_PATH = '/media/panlab/CHARVIHDD/SYDE671/DestripeNet'
IMAGE_PATH = ''
'''
DESTRIPE_DATA_CSV: path to data .csv file generated in noisy_img.py -> generate_destripe_data()
DESTRIPE_DATA_PATH: path to destripe training and validation folders 
                    (or folder where split_destripe() will generate training/validation folders)
MODEL_PATH: path to where model is saved during training
IMAGE_PATH: path to images for for destripe training
'''



def main():
    if SPLIT_DESTRIPE:
        split_destripe(DESTRIPE_DATA_CSV, DESTRIPE_DATA_PATH)

    destripe_path_train = os.path.join(DESTRIPE_DATA_PATH, 'train')
    destripe_path_val = os.path.join(DESTRIPE_DATA_PATH, 'val')

    train_loader, val_loader = destripe_loaders(
        os.path.join(destripe_path_train, 'train_inputs.csv'),
        os.path.join(destripe_path_train, 'train_labels.csv'),
        os.path.join(destripe_path_val, 'val_inputs.csv'),
        os.path.join(destripe_path_val, 'val_labels.csv'),
        IMAGE_PATH,
        batch_size=hp.D_BATCH_SIZE,
    )

    model = DestripeNet(
        in_channels=38,
        out_channels=1,
    ).to(hp.DEVICE)
    inspect_model(writer, model, train_loader, hp.DEVICE)

    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=hp.D_LEARNING_RATE)

    destripe_train_fn(
        train_loader,
        val_loader,
        model,
        optimizer,
        loss_fn,
        hp.D_NUM_EPOCH,
        hp.DEVICE,
        MODEL_PATH,
        writer,
        val_interval=2,
    )
    
if __name__ == '__main__':
    main()