'''
generate noisy images for training
'''
import numpy as np
import os
import cv2 as cv
import random
from parfor import parfor
import decompand
from planetaryimageEDR import PDS3ImageEDR
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def non_linearity(image):
    '''
    add non-linearity response noise, leaving parameters a, b, c, and d as placeholder
    :param image: numpy array of dimensions (MxN)
    :return: img of same dimensions with non-linear response noise
    '''
    img = np.zeros(image.shape)
    return img


def rescale_DN(image):
    '''
    Rescale images to be between 0-60 DN
    :param image:
    :return:
    '''
    multiply_factor = np.random.randint(1, 61)
    img = image/np.median(image) * multiply_factor
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


def generate_destripe_params(dark_calibration_folder, destination_folder, summed):
    '''
    :param dark_calibration_folder:
    :param destination_folder:
    :param summed: bool if using summed or normal images
    :return:
    '''
    param_destination = destination_folder + 'parameters'
    calib_destination = destination_folder + 'calibration_frames'
    dark_files = os.listdir(dark_calibration_folder)

    parameters = []

    for i in range(len(dark_files)):
        I = PDS3ImageEDR.open(os.path.join(dark_calibration_folder, dark_files[i]))
        labels = I.label

        if summed:
            masked_pix1 = list(I.image[0, :11])
            masked_pix2 = list(I.image[-1, -19:])
        else:
            masked_pix1 = list(I.image[0, :21])
            masked_pix2 = list(I.image[-1, -39:])
            
        parameters.append(
            {
                'Filename': dark_files[i],
                'Orbit_num': labels['ORBIT_NUMBER'],
                'FPGA_temp': labels['LRO:TEMPERATURE_FPGA'][0],
                'CCD_temp': labels['LRO:TEMPERATURE_FPA'][0],
                'Tele_temp': labels['LRO:TEMPERATURE_TELESCOPE'][0],
                'SCS_temp': labels['LRO:TEMPERATURE_SCS'][0],
                'DAC_offset': [
                    labels['LRO:DAC_RESET_LEVEL'],
                    labels['LRO:CHANNEL_A_OFFSET'],
                    labels['LRO:CHANNEL_B_OFFSET'],
                ],
                'Masked_pix': masked_pix1 + masked_pix2 
            }
        )
        
        df = pd.DataFrame(parameters)
        df.to_csv(os.path.join(param_destination, 'dark_summed_parameters.csv'), index=False)


def generate_crop_list(input_img_dir, total_num_crops, max_img_crops=50, img_dims=(52224, 2532), destination_txt_file='crop_list.txt', crop_size=256):
    input_files = os.listdir(input_img_dir)
    
    img_l = img_dims[0]
    img_h = img_dims[1]
    crop_list = []
    i = 0

    while i < total_num_crops:
        img_idx = np.random.randint(0, len(input_files))
        img_num_crops = np.random.randint(1, max_img_crops)
        crop_l = np.random.randint(0, img_l - crop_size, img_num_crops)
        crop_h = np.random.randint(0, img_h - crop_size, img_num_crops)
        crop_list.append([img_idx, crop_h, crop_l])
        i += img_num_crops

    # @parfor(range(num_crops))
    # def generate_crops(i):
    #     # num of crops to be generated
    #     img_idx = random.randint(1,len(input_files))
    #     crop_l = random.randint(0,img_l - crop_size)
    #     crop_h = random.randint(0,img_h - crop_size)
    #     # crop_list.append([img_idx, [crop_h,crop_l]])
    #     crop = [img_idx, [crop_h, crop_l]]
    #
    #     return crop
    
    # crop_list = str(generate_crops)

    with open(destination_txt_file, "w") as output:
        output.write(str(crop_list))

    return crop_list
    



def generate_noisy_img_pairs(clean_img_dir, destination_dir, crop_list, flatfield_img_path, calibration_frame_dir):
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
    
    # set destination directories
    noisy_destination = destination_dir + 'noisy'
    clean_destination = destination_dir + 'clean'
    
    # get clean_files
    clean_files = os.listdir(clean_img_dir)
    
    window_size = 256

    @parfor(len(crop_list), (crop_list,))
    def noisy_img_pairs(i, c_list):
        
        image_index = c_list[0]
        crops = c_list[1]
        
        # get image
        image = cv.imread(clean_files[image_index])
        
        # get crops
        clean_crops = [image[crop:crop+window_size] for crop in crops]
        noisy_crops = []
        
        for patch in clean_crops:
            # get each noise component
            N = non_linearity(patch)
            F = flatfield(patch, flatfield_img_path)
            S_N_p = photon_noise(patch)
            D = dark_noise(calibration_frame_dir) # do i need the directory here?
            N_c = companding_noise(patch)
        
            # generate noisy patch
            noisy_patch = np.matmul(N, np.matmul(F, S_N_p)) + D + N_c
            
            noisy_crops.append(noisy_patch)
        
        
        
        os.save(noisy_crops, noisy_destination)
        os.save(clean_crops, clean_destination)
        
        return

