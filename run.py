import os
import torch.optim as optim
import hyperparameters as hp
import torch.nn as nn

from utils import (
    split_destripe,
    destripe_loaders,
    photon_loaders,
)

from train import (
    destripe_train_fn,
    #photon_train_fn,
)

from destripe import DestripeNet

SPLIT_DESTRIPE = False

# Paths
DESTRIPE_DATA_CSV = 'test_running\dark_summed_data.csv'
DESTRIPE_DATA_PATH = 'test_running'
MODEL_PATH = 'test_running'



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
        batch_size=hp.BATCH_SIZE,
    )

    model = DestripeNet(
        in_channels=38,
        out_channels=1,
    ).to(hp.DEVICE)

    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=hp.LEARNING_RATE)

    destripe_train_fn(
        train_loader,
        val_loader,
        model,
        optimizer,
        loss_fn,
        hp.NUM_EPOCH,
        hp.DEVICE,
        MODEL_PATH,
        val_interval=2,
    )
    
if __name__ == '__main__':
    main()