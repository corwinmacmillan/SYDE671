'''
generate noisy images for training
'''
import numpy as np
import os
import cv2 as cv
import decompand
import random
import parfor


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
    img = np.zeros(image.shape)
    return img


def dark_noise(calibration_frame_dir):
    '''
    Return random window of dark calibration frame, chosen from director of cropped calibration frames
    :param calibration_frame_dir:
    :return: random calibration frame window
    '''
    files = os.listdir(calibration_frame_dir)
    rand_i = np.random.randint(0, len(files))
    rand_img = cv.imread(files[rand_i])
    return rand_img


def companding_noise(image):
    '''
    Add companding noise
    :param image:
    :return:
    '''
    img = decompand.compand(image, 3)
    img = decompand.decompand(img, 3)
    return img


def generate_destripe_pairs(dark_calibration_folder, destination_folder):
    '''

    :param dark_calibration_folder:
    :param destination_folder:
    :return:
    '''
    param_destination = destination_folder + 'parameters'
    calib_destination = destination_folder + 'calibration_frames'
    dark_files = os.listdir(dark_calibration_folder)


def generate_crop_list(clean_img_dir):
    clean_files = os.listdir(clean_img_dir)
    
    img_len = 2532
    crop_list = []
    
    for i in range(1e5):
        # num of crops to be generated
        img_idx = random.randint(1,len(clean_files))
        crop = random.randint(0,img_len - 256)
        crop_list.append([img_idx, crop])
    
        return(crop_list)



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

    
    for image_index, crops in crop_list:
        
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

