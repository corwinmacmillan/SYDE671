import os
import torch.optim as optim
import torch
# from models import hyperparameters as hp
import torch.nn as nn
import models.hyperparameters as hp

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
writer = SummaryWriter('dataset_creation/tensorboard')

# Conditional functions in main()
SPLIT_DESTRIPE = False
'''
SPLIT_DESTRIPE: split the data .csv file generated in noisy_img.py -> 
                generate_destripe_data() into training and validation folders
'''

# Paths
DESTRIPE_DATA_CSV = ''
DESTRIPE_DATA_PATH = r'D:\Jonathan\3_Courses\DestripeNet\NAC_L'
MODEL_PATH = r'D:\Jonathan\3_Courses\DestripeNet\NAC_L\model'
IMAGE_PATH = r'D:\Jonathan\3_Courses\dark_summed\NAC_L'
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
        os.path.join(destripe_path_train, 'train_inputs_reduced.csv'), # CHANGE FROM REDUCED
        os.path.join(destripe_path_train, 'train_labels_reduced.csv'),
        os.path.join(destripe_path_val, 'val_inputs_reduced.csv'),
        os.path.join(destripe_path_val, 'val_labels_reduced.csv'),
        IMAGE_PATH,
        batch_size=hp.D_BATCH_SIZE,
    )

    model = DestripeNet(
        in_channels=38,
        out_channels=1,
    ).to(hp.DEVICE)
    #inspect_model(writer, model, train_loader, hp.DEVICE)

    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=hp.D_LEARNING_RATE)

    # if os.listdir(MODEL_PATH) is not None:
    #     checkpoint_file = os.listdir(MODEL_PATH)
    #     checkpoint = torch.load(checkpoint_file)
    #     model.load_state_dict(checkpoint['model_state_dict'])
    #     optimizer.load_state_dict(checkpoint['optim_state_dict'])
    #     # epoch = checkpoint['epoch']
    #     # loss = checkpoint['train_loss']

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
    )
    
if __name__ == '__main__':
    main()