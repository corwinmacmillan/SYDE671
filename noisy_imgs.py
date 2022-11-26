'''
generate noisy images for training
'''
import numpy as np
import os
import cv2 as cv
import decompand
from planetaryimageEDR import PDS3ImageEDR
import pandas as pd
import sys


def non_linearity(image):
    '''
    add non-linearity response noise, leaving parameters a, b, c, and d as placeholder
    :param image: numpy array of dimensions (MxN)
    :return: img of same dimensions with non-linear response noise
    '''
    img = np.zeros(image.shape)
    return img


def flatfield(image, flatfield_img_path):
    '''
    add flatfield response noise, leaving the image path to base flatfield response off of as placeholder
    :param image: numpy array of dimensions (MxN)
    :param flatfield_img_path: path to image(s?) to generate flatfield response noise
    :return: img of same dimensions with flatfield noise
    '''
    img = np.zeros(image.shape)
    return img


def photon_noise(image):
    '''
    add photon noise
    :param image: numpy array of dimensions (MxN)
    :return:
    '''
    mean = np.mean(image)
    Np = np.random.poisson(mean, size=image.shape)
    return Np


def dark_noise(calibration_frame_dir, patch_size=(256,256)):
    '''
    Return random window of dark calibration frame, chosen from director of cropped calibration frames
    :param calibration_frame_dir:
    :return: random calibration frame window
    '''
    from sklearn.feature_extraction.image import extract_patches_2d
    files = os.listdir(calibration_frame_dir)
    rand_i = np.random.randint(0, len(files))
    rand_img = PDS3ImageEDR.open(files[rand_i])
    rand_patch = extract_patches_2d(rand_img, patch_size)
    return rand_patch


def companding_noise(image):
    '''
    Add companding noise
    :param image:
    :return:
    '''
    img = decompand.compand(image, 3)
    img = decompand.decompand(img, 3)
    return img


def generate_destripe_data(dark_calibration_folder, destination_folder, summed):
    '''
    :param dark_calibration_folder:
    :param destination_folder:
    :param summed: bool if using summed or normal images
    :return:
    '''
    param_destination = os.path.join(destination_folder, 'parameters')
    #dark_line_destination = os.path.join(destination_folder, 'calibration_lines')
    dark_files = os.listdir(dark_calibration_folder)

    parameters = []
    np.set_printoptions(threshold=sys.maxsize)

    for i in range(len(dark_files)):
        I = PDS3ImageEDR.open(os.path.join(dark_calibration_folder, dark_files[i]))
        labels = I.label

        for j in range(I.image.shape[0]):
            line = I.image[i, :]

            if summed:
                masked_pix1 = list(line[:11])
                masked_pix2 = list(line[-19:])
            else:
                masked_pix1 = list(line[:21])
                masked_pix2 = list(line[-39:])
                
            parameters.append(
                {
                    'Filename': dark_files[i],
                    'Orbit_num': labels['ORBIT_NUMBER'],
                    'FPGA_temp': labels['LRO:TEMPERATURE_FPGA'][0],
                    'CCD_temp': labels['LRO:TEMPERATURE_FPA'][0],
                    'Tele_temp': labels['LRO:TEMPERATURE_TELESCOPE'][0],
                    'SCS_temp': labels['LRO:TEMPERATURE_SCS'][0],
                    'DAC_reset': labels['LRO:DAC_RESET_LEVEL'],
                    'DAC_A': labels['LRO:CHANNEL_A_OFFSET'],
                    'DAC_B': labels['LRO:CHANNEL_B_OFFSET'],
                    'Masked_pix': masked_pix1 + masked_pix2,
                    'Pixel_line': line, 
                }
            )
        
    df = pd.DataFrame(parameters)
    if summed:
        df.to_csv(os.path.join(param_destination, 'dark_summed_data.csv'), index=False)
    else:
        df.to_csv(os.path.join(param_destination, 'dark_normal_data.csv'), index=False)


def generate_crop_list(clean_img_dir):
    clean_files = os.listdir(clean_img_dir)


def generate_noisy_img_pairs(clean_img_dir, destination_dir, crop_list):
    '''
    Generate clean-noisy image pairs. Let's make this parallelized
    Workflow:
        noisy_dir = destination_dir + 'noisy'
        clean_destination = destination_dir + 'clean'
        clean_files = os.listdir(clean_img_dir)
        parfor image_index, crops in crop_list: #parfor is parallelized for loop
            image = cv.imread(clean_files[image_index])
            noise = Sequential(non_linearity, flatfield, photon_noise)
            noisy_image = noise(image)
            clean_crops = [image[crop:crop+window_size] for crop in crops]
            noisy_crops = [noisy_image """"""]
            noisy_crops = [dark_noise + compand_noise(crop) for crop in crop_images]
            save(noisy_crops, noisy_dir) #  make sure corresponding
            save(clean_crops, clean_dir) #  crops have corresponding names!!

    :param clean_img_dir:
    :param noisy_img_dir:
    :return:
    '''
    noisy_destination = destination_dir + 'noisy'
    clean_destination = destination_dir + 'clean'
    clean_files = os.listdir(clean_img_dir)

