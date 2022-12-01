import torch
import os
import numpy as np
import pandas as pd

from models.destripe import DestripeNet

from torch.utils.data import (DataLoader, Dataset)

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter(log_dir='dataset_creation/tensorboard')


from dataset_creation.dataset import Destripe_Dataset

from utils.util import(L1_loss)

import models.hyperparameters as hp

from utils.planetaryimageEDR import PDS3ImageEDR

TEST_DESTRIPE = True


DESTRIPE_DATA_PATH = r'D:\Jonathan\3_Courses\DestripeNet\NAC_R'
IMAGE_PATH = r'D:\Jonathan\3_Courses\dark_summed\NAC_R'
MODEL_PATH = r'D:\Jonathan\3_Courses\DestripeNet\NAC_R\model'
PSR_PATH = ''
'''
DESTRIPE_DATA_PATH: path to destripe testing folders
                    (or folder where split_destripe() will generate training/validation/testing folders)
MODEL_PATH: path to where best model is saved
MODEL_PATH: path to where best model is saved# 
'''

BATCH_SIZE = 32
NUM_WORKERS = 8


def test_destripe():
    destripe_path_test = os.path.join(DESTRIPE_DATA_PATH, 'test')

    test_ds = Destripe_Dataset(
        input_file=os.path.join(destripe_path_test, 'test_inputs.csv'),
        label_file=os.path.join(destripe_path_test, 'test_labels.csv')
    )

    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=NUM_WORKERS)

    model = DestripeNet(
        in_channels=38,
        #out_channels=1,
    ).to(hp.DEVICE)
    model.load_state_dict(torch.load(
        os.path.join(MODEL_PATH, 'best_L1_model.pth'),
        map_location='cpu'
    ))
    
    test_L1_values = []
    model.eval
    with torch.no_grad():
        test_L1 = 0
        test_L1_run = 0

        for step, (test_inputs, test_labels) in enumerate(test_loader):

            test_inputs, test_labels = (test_inputs.to(hp.DEVICE), test_labels.to(hp.DEVICE))

            test_outputs = model(test_inputs)

            # L1 loss
            test_L1 = L1_loss(test_outputs, test_labels)
            test_L1_run += test_L1
            writer.add_scalar('Test L1 Loss', test_L1_run/(step+1), (step+1))
            test_L1_values.append(test_L1_run/(step+1))
            if (step+1) % 100 == 0:
                print('Step: {}/{} \tTest L1 Loss: {}'.format(
                    (step+1),
                    len(test_loader), 
                    test_L1_run / (step+1))
                )
    
    df = pd.DataFrame(test_L1_values)
    df.to_csv('dataset_creation/tensorboard/test_L1_values.csv', header=False, index=False)
            

class PSR_Destripe(Dataset):
    '''
    PSR DestripeNet dataset
    :params:
        self.X: raw data .csv file generated from generate_destripe_data()
        self.transform: transforms
    '''
    def __init__(self, input_file, transform=None):
        # Read input and output csv
        self.X = pd.read_csv(input_file)
        self.transform = transform
    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        input_data = self.X.iloc[index, 2:].to_numpy()
        mask_pix = np.fromstring(input_data[-1][1:-1], sep=',')
        input = np.concatenate((input_data[:-1], mask_pix)).astype(np.float32)
        input = input.reshape(38, 1)
        
        label_data = self.y.iloc[index, 2:].to_numpy()[0]
        label_data = re.sub('\n', '', label_data)
        label = np.fromstring(label_data[1:-1], sep=' ').astype(np.float32)

        if self.transform is not None:
            input = self.transform(input)
            label = self.transform(label)

        return input, label

def PSR_destripe():
    PSR_files = os.listdir(PSR_PATH)

    PSR_data = []
    for i in range(len(PSR_files)):
        I = PDS3ImageEDR.open(os.path.join(PSR_PATH, PSR_files[i]))
        labels = I.label
        for j in range(I.image.shape[0]):
            line = I.image[j, :]
  
            masked_pix1 = list(line[:11])
            masked_pix2 = list(line[-19:])
    
            PSR_parameters = np.array(
                PSR_files[i],
                labels['ORBIT_NUMBER'],
                labels['LRO:TEMPERATURE_FPGA'][0],
                labels['LRO:TEMPERATURE_FPA'][0],
                labels['LRO:TEMPERATURE_TELESCOPE'][0],
                labels['LRO:TEMPERATURE_SCS'][0],
                labels['LRO:DAC_RESET_LEVEL'],
                labels['LRO:CHANNEL_A_OFFSET'],
                labels['LRO:CHANNEL_B_OFFSET'],
            )
            PSR_parameters = np.concatenate(PSR_parameters, masked_pix1, masked_pix2)
            PSR_data.append(PSR_parameters)


        PSR_ds = 'INCOMPLETE'

        PSR_loader = DataLoader()

        model = DestripeNet(
            in_channels=38,
            out_channels=1,
        ).to(hp.DEVICE)
        model.load_state_dict(torch.load(
            os.path.join(MODEL_PATH, 'best_L1_model.pth')
        ))
        
        test_L1_values = []
        model.eval
        with torch.no_grad():
            

            for (test_inputs, test_labels) in test_loader:

                test_inputs, test_labels = (test_inputs.to(hp.DEVICE), test_labels.to(hp.DEVICE))

                test_outputs = model(test_inputs)

          
def main():
    if TEST_DESTRIPE:
        test_destripe()

    

if __name__ == '__main__':
    main()