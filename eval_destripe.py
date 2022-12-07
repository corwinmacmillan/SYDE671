import torch
import os
import numpy as np
import pandas as pd
import re
from parfor import parfor

from models.destripe import DestripeNet

from torch.utils.data import (DataLoader, Dataset)

import models.hyperparameters as hp

from utils.planetaryimageEDR import PDS3ImageEDR

# DESTRIPE_DATA_PATH = r'D:\Jonathan\3_Courses\DestripeNet\NAC_R'
# IMAGE_PATH = r'D:\Jonathan\3_Courses\dark_summed\NAC_R'
MODEL_PATH = '/media/panlab/EXTERNALHDD/DestripeNet/NAC_L/model/best'
NOISY_PATH = '/media/panlab/CHARVIHDD/PhotonNet/NAC_L/test/noisy'
PSR_OUTPUT_PATH = '/media/panlab/CHARVIHDD/PhotonNet/NAC_L/test/destripe_output'
CROP_FILE = 'dataset_creation/summed_L_completed_crops.txt'
CROP_PATH = '/media/panlab/CHARVIHDD/PhotonNet/NAC_L/test/destripe_crops'

'''
DESTRIPE_DATA_PATH: source path to destripe testing folders
                    (or folder where split_destripe() will generate training/validation/testing folders)
MODEL_PATH: source path to where best model is saved
PSR_PATH: source path to PSR images
PSR_OUTPUT_PATH: destination path to outputs of destripe PSR images
CROP_FILE = .txt tiles with filtered crop file coordinates
CROP_PATH = destination path to save crop
'''

BATCH_SIZE = 32
NUM_WORKERS = 16
PATCH_SIZE = 256

class PSR_Destripe(Dataset):
    '''
    PSR DestripeNet dataset
    :params:
        self.X: raw data .csv file generated from generate_destripe_data()
    '''
    def __init__(self, image_data):
        # get input data
        self.image_data = image_data
    def __len__(self):
        return len(self.image_data)

    def __getitem__(self, index):
        input = self.image_data[index].reshape(38, 1)

        return input # Return an array for each line of the PSR


def eval_destripe():
    PSR_files = os.listdir(NOISY_PATH)

    for i in range(len(PSR_files)):
    # @parfor(range(len(PSR_files)))
    # def generate(i):
        print('=' * 30)
        print('Evaluating Image {}'.format(i+1))

        I = PDS3ImageEDR.open(os.path.join(NOISY_PATH, PSR_files[i]))
        label = I.label
        image = I.image
        # with open(os.path.join(NOISY_PATH, PSR_files[i]), 'rb') as f:
        #     image = np.fromfile(f, dtype=np.uint16).reshape(52224, 2532)

        PSR_data = [] # Input data for destripe
        for j in range(image.shape[0]): # for each line in the image
            line = image[j, :]
  
            masked_pix1 = line[:11]
            masked_pix2 = line[-19:]

            PSR_parameters = np.array([
                label['ORBIT_NUMBER'],
                label['LRO:TEMPERATURE_FPGA'][0],
                label['LRO:TEMPERATURE_FPA'][0],
                label['LRO:TEMPERATURE_TELESCOPE'][0],
                label['LRO:TEMPERATURE_SCS'][0],
                label['LRO:DAC_RESET_LEVEL'],
                label['LRO:CHANNEL_A_OFFSET'],
                label['LRO:CHANNEL_B_OFFSET'],
            ])
            PSR_parameters = np.concatenate((PSR_parameters, masked_pix1, masked_pix2))
            PSR_data.append(PSR_parameters)


        PSR_ds = PSR_Destripe(PSR_data)

        PSR_loader = DataLoader(PSR_ds, batch_size=BATCH_SIZE, pin_memory=True, num_workers=NUM_WORKERS)

        model = DestripeNet(
            in_channels=38,
            # out_channels=1,
        ).to(hp.DEVICE)
        model.load_state_dict(torch.load(
            os.path.join(MODEL_PATH, 'best_L1_model.pth'),
            map_location='cpu'
        ))
        
        model.eval()
        with torch.no_grad():

            output_image = np.zeros((52224, 2532)).astype(np.uint16)
            count = 0
            for step, line in enumerate(PSR_loader):
                
                line = (line.float().to(hp.DEVICE))

                line_out = model(line)

                for j in range(BATCH_SIZE):
                    output_image[count+j, :] = line_out.cpu().detach().numpy()[j].astype(np.uint16)
                
                if step+1 % 200 == 0:
                    print('Step: {}/{}'.format((step+1), len(PSR_loader)))
                
                count += BATCH_SIZE

        print('Saving file...')
        with open(os.path.join(PSR_OUTPUT_PATH, 'Destripe_' + PSR_files[i]), 'wb') as f:
            f.write(bytes(output_image))
        print('Save Complete')


def crop_PSR():
    with open(CROP_FILE, 'r') as f1:
        df = pd.DataFrame(map(eval, f1.read().splitlines()))
    f1.close()

    PSR_destripe_files = os.listdir(PSR_OUTPUT_PATH)

    for i in range(len(PSR_destripe_files)):
        with open(os.path.join(PSR_OUTPUT_PATH, PSR_destripe_files[i]), 'rb') as f2:
            dark_noise = np.fromfile(f2, dtype=np.uint16).reshape(52224, 2532)

        filename = PSR_destripe_files[i].split('_')[1]

        for j in range(len(df)):
            if df.iloc[j, 0] == filename:
                crop_row = df.iloc[j, 1]
                crop_col = df.iloc[j, 2]
                crop = dark_noise[
                    crop_row : (crop_row + PATCH_SIZE),
                    crop_col : (crop_col + PATCH_SIZE)
                ]

                with open(os.path.join(CROP_PATH, '{}_{}_{}'.format(crop_row, crop_col, filename)), 'wb') as f:
                    f.write(bytes(crop))


def main():
    eval_destripe()
    # crop_PSR()


if __name__ == '__main__':
    main()

