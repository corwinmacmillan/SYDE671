import torch
import os
import numpy as np
import pandas as pd
import re

from models.destripe import DestripeNet

from torch.utils.data import (DataLoader, Dataset)

import models.hyperparameters as hp

from utils.planetaryimageEDR import PDS3ImageEDR

# DESTRIPE_DATA_PATH = r'D:\Jonathan\3_Courses\DestripeNet\NAC_R'
# IMAGE_PATH = r'D:\Jonathan\3_Courses\dark_summed\NAC_R'
MODEL_PATH = r'D:\Jonathan\3_Courses\DestripeNet\NAC_R\model'
PSR_PATH = r'C:\Users\jh3chu\OneDrive - University of Waterloo\SYDE 671\SYDE671\test_running\PSR'
PSR_OUTPUT_PATH = PSR_PATH
'''
DESTRIPE_DATA_PATH: path to destripe testing folders
                    (or folder where split_destripe() will generate training/validation/testing folders)
MODEL_PATH: path to where best model is saved
PSR_PATH: path to PSR images
PSR_OUTPUT_PATH: path to outputs of destripe PSR images
'''

BATCH_SIZE = 32
NUM_WORKERS = 8

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
    PSR_files = os.listdir(PSR_PATH)

    for i in range(len(PSR_files)):
        print('=' * 30)
        print('Evaluating Image {}'.format(i+1))

        I = PDS3ImageEDR.open(os.path.join(PSR_PATH, PSR_files[i]))
        label = I.label
        image = I.image

        PSR_data = [] # Input data for destripe
        for j in range(I.image.shape[0]): # for each line in the image
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
        
        model.eval
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
        with open(os.path.join(PSR_PATH, 'Destripe_' + PSR_files[i]), 'wb') as f:
            f.write(output_image)
        print('Save Complete')
        
def main():
    eval_destripe()

if __name__ == '__main__':
    main()