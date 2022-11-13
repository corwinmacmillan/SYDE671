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


def download_image(file_spec, filepath):
    img_name = file_spec[-16:]

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


def download_image_list(img_csv, download_path):
    img_list = pd.read_csv(img_csv)
    file_specs = img_list[['FILE_SPECIFICATION_NAME']].to_numpy()
    for file_spec in file_specs:
        download_image(file_spec[0], download_path)


if __name__ == '__main__':
    csv_dir = 'edr/'
    download_dir = '/media/panlab/EXTERNALHDD/'
    for file in os.listdir(csv_dir):
        download_image_list(csv_dir + file, download_dir + file[:-4] + '/')
       



       

       
    
    
    





