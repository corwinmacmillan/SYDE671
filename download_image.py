#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 13 14:47:35 2022

@author: charvichoudhary
"""

import requests
import numpy as np
import pandas as pd
import os
from pathlib import Path
import time
from parfor import parfor


def download_image(file_spec, filepath):
    img_name = file_spec[-16:]
    img_name = img_name.replace('3-CDR', '2-EDR')
    img_name = img_name.replace('C.IMG', 'E.IMG')
    img_name = img_name.replace('LRC_1', 'LRC_0')
    img_url = 'https://pds.lroc.asu.edu/data/' + file_spec
    # try:
    r = requests.get(img_url)
    # except:
    #     raise RuntimeError('Link {} does not exist'.format(img_url))
    Path(filepath).mkdir(parents=True, exist_ok=True)
    with open(filepath + img_name, 'wb') as f:
        try:
            f.write(r.content)
        except:
            raise RuntimeError('Directory is full. Stopped downloading at csv {}, product ID {}'.format(filepath,
                                                                                                        img_name))


def download_image_list(img_csv, download_path, continuation_index=0):
    img_list = pd.read_csv(img_csv)
    file_specs = img_list[['FILE_SPECIFICATION_NAME']].to_numpy()
    list_size = file_specs.size
    sum_time = 0
    for i, file_spec in enumerate(file_specs[continuation_index:]):
        start = time.time()
        download_image(file_spec[0], download_path)
        end = time.time()
        sum_time += end - start
        hours_left = round((sum_time/(i+1) * list_size - (i + continuation_index) * sum_time/(i+1))/3600, 2)
        print('File {}, {} / {}, estimated hours left: {}'.format(img_csv + file_spec[0], i + continuation_index,
                                                                  list_size, hours_left))


def separate_camera_modes(camera_dir):
    img_files = os.listdir(camera_dir)
    @parfor(len(img_files))
    def separate_files(i):
        img_file = img_files[i]
        if img_file[-6] == 'L':
            os.rename(os.path.join(camera_dir, img_file), os.path.join(camera_dir, 'NAC_L/', img_file))
        elif img_file[-6] == 'R':
            os.rename(os.path.join(camera_dir, img_file), os.path.join(camera_dir, 'NAC_R/', img_file))
        else:
            raise RuntimeError('File name does not match expected ISIS naming convention')


if __name__ == '__main__':
    csv_dir = 'edr/'
    download_dir = '/media/panlab/EXTERNALHDD/'
    for csv_file in os.listdir(csv_dir):
        if csv_file == 'bright_normal.csv' or csv_file == 'dark_normal.csv':
            pass
        else:
            download_image_list(csv_dir + csv_file, download_dir + csv_file[:-4] + '/', 34)
       



       

       
    
    
    





