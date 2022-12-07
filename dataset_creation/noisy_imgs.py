'''
generate noisy images for training
'''
import time

import numpy as np
import os
from parfor import parfor
from utils import decompand
from utils.planetaryimageEDR import PDS3ImageEDR
import pandas as pd
import sys
from utils.myisis import MyIsis

# import cv2

ISISROOT = "/media/panlab/EXTERNALHDD/data/lro/calibration/"


def non_linearity(image, mode_and_camera='summed_L', forward=True):
    '''
    add non-linearity response noise, leaving parameters a, b, c, and d as placeholder
    :param image: numpy array of dimensions (MxN)
    :return: img of same dimensions with non-linear response noise
    '''
    isis = MyIsis(ISISROOT)
    img = isis.nonlinearity(image, mode_and_camera, forward=forward)
    return img.astype(np.uint16)


def rescale_DN(image):
    '''
    Rescale images to be between 0-60 DN
    :param image:
    :return:
    '''
    multiply_factor = np.random.rand() * 60
    img = image / np.median(image) * multiply_factor
    return img.astype(np.uint16)


def flatfield(image, mode_and_camera='summed_R', line_slice_start=None, line_slice_end=None, forward=True):
    '''
    add flatfield response noise, leaving the image path to base flatfield response off of as placeholder
    :param mode_and_camera: string denoting NAC mode (summed, normal) and NAC Camera (L, R)
    :param line_slice_end: beginning column used for image crop
    :param line_slice_start: ending column used for image crop
    :param image: numpy array of dimensions (MxN)
    :param flatfield_img_path: path to image(s?) to generate flatfield response noise
    :return: img of same dimensions with flatfield noise
    '''
    isis = MyIsis(ISISROOT)
    img = isis.flatfield(image, k=mode_and_camera, line_slice_start=line_slice_start, line_slice_end=line_slice_end,
                         forward=forward)
    return img.astype(np.uint16)


def photon_noise(image):
    '''
    add photon noise
    :param image: numpy array of dimensions (MxN)
    :return:
    '''
    mean = np.mean(image)
    Np = np.random.poisson(mean, size=image.shape).astype(np.uint16)
    return image + Np


def dark_noise(calibration_frame_dir, patch_size=256):
    '''
    Return random window of dark calibration frame, chosen from director of cropped calibration frames
    :param calibration_frame_dir:
    :return: random calibration frame window
    '''
    # from sklearn.feature_extraction.image import extract_patches_2d
    files = os.listdir(calibration_frame_dir)
    rand_i = np.random.randint(0, len(files))
    rand_img = PDS3ImageEDR.open(calibration_frame_dir + files[rand_i]).image
    # rand_patch = extract_patches_2d(rand_img, patch_size)
    if patch_size is not None:
        img_dims = rand_img.shape
        crop_l = np.random.randint(0, img_dims[1] - patch_size)
        crop_h = np.random.randint(0, img_dims[0] - patch_size)
        rand_img = rand_img[crop_h:crop_h + patch_size, crop_l:crop_l + patch_size]
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


def generate_destripe_data(dark_calibration_folder, destination_folder, summed=True):
    '''
    :param dark_calibration_folder:
    :param destination_folder:
    :param summed: bool if using summed or normal images
    :return:
    '''
    # param_destination = os.path.join(destination_folder, 'DestripeNet')
    # dark_line_destination = os.path.join(destination_folder, 'calibration_lines')
    dark_files = os.listdir(dark_calibration_folder)

    # sum_time = 0
    headers = ['Filename',
               'Orbit_num',
               'FPGA_temp',
               'CCD_temp',
               'Tele_temp',
               'SCS_temp',
               'DAC_reset',
               'DAC_A',
               'DAC_B',
               'Masked_pix',
               'Pixel_line_index']
    headers = pd.DataFrame(columns=headers)
    if summed:
        headers.to_csv(os.path.join(destination_folder, 'dark_summed_data.csv'), index=False, mode='w')
    else:
        headers.to_csv(os.path.join(destination_folder, 'dark_normal_data.csv'), index=False, mode='w')

    @parfor(range(len(dark_files)))
    def generate_destripe(i):
    # for i in range(len(dark_files)):
    #     start = time.time()
        I = PDS3ImageEDR.open(os.path.join(dark_calibration_folder, dark_files[i]))
        np.set_printoptions(threshold=sys.maxsize)
        labels = I.label
        parameters = []
        for j in range(I.image.shape[0]):
            line = I.image[j, :]

            if summed:
                masked_pix1 = list(line[:11])
                masked_pix2 = list(line[-19:])
            else:
                masked_pix1 = list(line[:21])
                masked_pix2 = list(line[-39:])

            parameters.append([dark_files[i],
                               labels['ORBIT_NUMBER'],
                               labels['LRO:TEMPERATURE_FPGA'][0],
                               labels['LRO:TEMPERATURE_FPA'][0],
                               labels['LRO:TEMPERATURE_TELESCOPE'][0],
                               labels['LRO:TEMPERATURE_SCS'][0],
                               labels['LRO:DAC_RESET_LEVEL'],
                               labels['LRO:CHANNEL_A_OFFSET'],
                               labels['LRO:CHANNEL_B_OFFSET'],
                               masked_pix1 + masked_pix2,
                               j
                               ])

        df = pd.DataFrame(parameters)
        if summed:
            df.to_csv(os.path.join(destination_folder,  'dark_summed_data.csv'), header=False, index=False, mode='a')
        else:
            df.to_csv(os.path.join(destination_folder, 'dark_normal_data.csv'), header=False, index=False, mode='a')

        # end = time.time()
        # sum_time += end - start
        # hours_left = round((sum_time / (i + 1) * len(dark_files) - i * sum_time / (i + 1)) / 3600, 2)
        # print('Image {}, {} / {}, estimated hours left: {}'.format(dark_files[i], i, len(dark_files), hours_left))

def generate_crop_list(input_img_dir, total_num_crops, max_img_crops=50, img_dims=(52224, 2532),
                       destination_npy_file='crop_list.npy', crop_size=256):
    input_files = os.listdir(input_img_dir)

    img_l = img_dims[1]
    img_h = img_dims[0]
    crop_list = []
    i = 0

    while i < total_num_crops:
        img_idx = np.random.randint(0, len(input_files))
        img_num_crops = np.random.randint(1, max_img_crops)
        crop_l = np.random.randint(0, img_l - crop_size, img_num_crops)
        crop_h = np.random.randint(0, img_h - crop_size, img_num_crops)
        crop_list.append([img_idx, crop_h, crop_l])
        i += img_num_crops

    np.save(destination_npy_file, crop_list, allow_pickle=True)

    return crop_list


def generate_noisy_img_pairs(clean_img_dir, destination_dir, crop_list, calibration_frame_dir, mode_and_camera,
                             crop_size=256):
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
    noisy_destination = os.path.join(destination_dir, 'noisy/')
    clean_destination = os.path.join(destination_dir, 'clean/')

    # get clean_files
    clean_files = os.listdir(clean_img_dir)

    # @parfor(range(crop_list.shape[0]), (crop_list,))
    def noisy_img_pairs(i, c_list):

        image_index = c_list[i, 0]
        crops_h = c_list[i, 1]
        crops_l = c_list[i, 2]

        # get image
        img_file = clean_files[image_index]
        image = PDS3ImageEDR.open(os.path.join(clean_img_dir, img_file)).image
        # get crops
        for crop_h, crop_l in zip(crops_h, crops_l):
        # @parfor(range(len(crops_l)), (crops_h, ), (crops_l, ))
        # def add_noise(i, crop_h, crop_l):
            clean_crop = image[crop_h:crop_h + crop_size, crop_l:crop_l + crop_size]
            if np.median(clean_crop) > 200:
                clean_crop = rescale_DN(clean_crop)
                # get each noise component
                noisy_patch = photon_noise(clean_crop)
                noisy_patch = flatfield(noisy_patch, mode_and_camera=mode_and_camera,
                                        line_slice_start=crop_l, line_slice_end=crop_l + crop_size)
                noisy_patch = non_linearity(noisy_patch, mode_and_camera=mode_and_camera)
                noisy_patch += dark_noise(calibration_frame_dir)
                noisy_patch = companding_noise(noisy_patch)
                with open(os.path.join(clean_destination, '{}_{}_'.format(crop_h, crop_l) + img_file), 'wb') as clean:
                    clean.write(bytes(clean_crop))
                with open(os.path.join(noisy_destination, '{}_{}_'.format(crop_h, crop_l) + img_file), 'wb') as noisy:
                    noisy.write(bytes(noisy_patch))

                crop_dict = {'image': img_file, 'crop_row': crop_h, 'crop_col': crop_l}
                with open(mode_and_camera + '_completed_crops.txt', 'a') as f:
                    f.write(str(crop_dict) + '\n')

        # end = time.time()
        # sum_time += end - start
        # hours_left = round((sum_time / (i + 1) * len(dark_files) - i * sum_time / (i + 1)) / 3600, 2)
        # print('Image {}, {} / {}, estimated hours left: {}'.format(dark_files[i], i, len(dark_files), hours_left))

    for i in range(len(crop_list)):
        noisy_img_pairs(i, crop_list)


def generate_whole_noisy(clean_img_dir, destination_dir, calibration_frame_dir, mode_and_camera, number_whole_img=50):
    noisy_destination = os.path.join(destination_dir, 'noisy/')
    clean_destination = os.path.join(destination_dir, 'clean/')

    # get clean_files
    clean_files = np.random.choice(os.listdir(clean_img_dir), size=number_whole_img)
    # for img_file in clean_files:
    @parfor(range(number_whole_img))
    def generate(i):
        img_file = clean_files[i]
        I = PDS3ImageEDR.open(os.path.join(clean_img_dir, img_file))
        image = I.image
        if np.median(image) > 200:
            image = rescale_DN(image)
            # get each noise component
            noisy_img = photon_noise(image)
            noisy_img = flatfield(noisy_img, mode_and_camera=mode_and_camera)
            noisy_img = non_linearity(noisy_img, mode_and_camera=mode_and_camera)
            noisy_img += np.repeat(dark_noise(calibration_frame_dir, patch_size=None), 51, axis=0)
            noisy_img = companding_noise(noisy_img)
            I.save(os.path.join(clean_destination, img_file))
            I.update_image(decompand.compand(noisy_img, 3))
            I.save(os.path.join(noisy_destination, img_file))


def generate_lines_for_destripe(dark_calibration_folder, destination_folder):
    # import xarray as xr
    # import dask
    images = os.listdir(dark_calibration_folder)
    # destination_zarr = os.path.join(destination_folder, 'lines.zarr')
    chunk_size = 64
    sum_time = 0
    i = 0
    while i < len(images):
        start = time.time()
        if i + chunk_size >= len(images):
            chunk_size = len(images) % chunk_size
        images_chunk = np.zeros((chunk_size, 1024, 2532), dtype=np.uint16)

        # for j in range(chunk_size):
        @parfor(range(chunk_size))
        def generate(j):
            image_path = os.path.join(dark_calibration_folder, images[i + j])
            images_chunk[j] = PDS3ImageEDR.open(image_path).image
        for j in range(chunk_size):
            print('image {}/{} in batch'.format(j + 1, chunk_size))
            for k in range(1024):
                with open(os.path.join(destination_folder, images[j][:-4] + '_{}.IMG'.format(k)), 'wb') as f:
                    f.write(images_chunk[j, k, :])
        # da = xr.DataArray(data=images_chunk,
        #                         dims=['x', 'y', 'image_name'],
        #                         coords={
        #                             'x': range(1024),
        #                             'y': range(2532),
        #                             'image_name': images[i:i+chunk_size]
        #                         })
        # da = da.chunk(chunks={'x': 1, 'y': -1, 'image_name': 1})
        # if os.path.exists(destination_zarr):
        #     da.to_dataset(name='image').to_zarr(destination_zarr, append_dim='image_name')
        # else:
        #     da.to_dataset(name='image').to_zarr(destination_zarr)
        i += chunk_size
        end = time.time()
        sum_time += end - start
        print('image {}/{}, time left: {}'.format(i, len(images), ((sum_time / i) * len(images)) / 3600))



if __name__ == '__main__':
    # source_dir = '/media/panlab/EXTERNALHDD/bright_summed/NAC_L/'
    # generate_crop_list(source_dir, 100, destination_npy_file='test_crop_list_L.npy')
    # destination_dir = '/media/panlab/CHARVIHDD/PhotonNet/NAC_L/test/'
    calib_dir = '/media/panlab/EXTERNALHDD/dark_summed/NAC_L/images/'
    destination_dir = '/media/panlab/EXTERNALHDD/dark_summed/NAC_L/lines/'
    # lines_csv_dirs = '/media/panlab/EXTERNALHDD/DestripeNet/NAC_L'
    generate_lines_for_destripe(calib_dir, destination_dir)
    # for dir in os.listdir(lines_csv_dirs):
    #     if dir == 'test' or dir == 'train' or dir == 'val':
    #         lines_csv = os.path.join(lines_csv_dirs, dir, dir + '_labels_reduced.csv')
    #         generate_lines_for_destripe(lines_csv, calib_dir, destination_dir)
    # crop_list_path = np.load('test_crop_list_L.npy', allow_pickle=True)
    # generate_whole_noisy(source_dir, destination_dir, calib_dir, 'summed_L')

    # for sub_dir in os.listdir(source_dir):
    #     crop_list = np.load('crop_list_' + sub_dir[-1] + '.npy', allow_pickle=True)
    #     generate_noisy_img_pairs(source_dir + sub_dir, destination_dir + sub_dir,
    #                              crop_list, calib_dir + '/' + sub_dir + '/', 'summed_' + sub_dir[-1])
    # dark_calibration = '/media/panlab/EXTERNALHDD/dark_summed/'
    # generate_destripe_data(dark_calibration, destination_dir)
