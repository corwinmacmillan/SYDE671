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


def download_image(file_spec, filepath):
    # product_id = 'M1103765284LE'

    img_url = 'https://pds.lroc.asu.edu/data/' + file_spec
    # try:
    r = requests.get(img_url)
    # except:
    #     raise RuntimeError('Link {} does not exist'.format(img_url))

    with open(filepath + file_spec[-16:], 'wb') as f:
        try:
            f.write(r.content)
        except:
            raise RuntimeError('Directory is full. Stopped downloading at csv {}, product ID {}'.format(filepath,
                                                                                                        file_spec[-16:]))


def download_image_list(img_csv, download_path):
    img_list = pd.read_csv(img_csv)
    file_specs = img_list[['FILE_SPECIFICATION_NAME']].to_numpy()
    for file_spec in file_specs:
        download_image(file_spec, download_path)


if __name__ == '__main__':
    csv_dir = 'edr/'
    for file in os.listdir(csv_dir):
        download_image_list(csv_dir + file, '/media/panlab/EXTERNALHDD')
       



       

       
    
    
    





