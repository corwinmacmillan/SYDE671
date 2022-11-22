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
    for i, file_spec in enumerate(file_specs[continuation_index:]):
        download_image(file_spec[0], download_path)
        print('File {}, {} / {}'.format(img_csv + file_spec[0], i + continuation_index, list_size))


if __name__ == '__main__':
    csv_dir = 'edr/'
    download_dir = '/media/panlab/EXTERNALHDD/'
    for csv_file in [os.listdir(csv_dir)[1]]:
        download_image_list(csv_dir + csv_file, download_dir + csv_file[:-4] + '/')
       



       

       
    
    
    





