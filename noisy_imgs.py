'''
generate noisy images for training
'''
import numpy as np
import os
import cv2 as cv
import decompand


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


def dark_noise(image, calibration_frame_dir):
    '''
    Add dark noise by randomly sampling dark calibration frames and adding it to image
    :param image:
    :param calibration_frame_dir:
    :return:
    '''
    files = os.listdir(calibration_frame_dir)
    rand_i = np.random.randint(0, len(files))
    rand_img = cv.imread(files[rand_i])
    img = image + rand_img
    return img


def companding_noise(image):
    '''
    Add companding noise
    :param image:
    :return:
    '''
    img = decompand.compand(image, 3)
    img = decompand.decompand(img, 3)
    return img


def generate_noisy_img_pairs(clean_img_dir, destination_dir):
    '''
    Generate clean-noisy image pairs. Let's make this parallelized
    :param clean_img_dir:
    :param noisy_img_dir:
    :return:
    '''
    noisy_destination = destination_dir + 'noisy'
    clean_destination = destination_dir + 'clean'
    clean_files = os.listdir(clean_img_dir)