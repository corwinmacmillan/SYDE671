#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 13 14:47:35 2022

@author: charvichoudhary
"""

import requests
import pandas as pd
import numpy as np


def download_image(product_id, filepath):
    # product_id = 'M1103765284LE'
    
    try:
        img_url = 'https://pds.lroc.asu.edu/data/LRO-L-LROC-2-EDR-V1.0/LROLRC_0013/DATA/ESM/2012275/NAC/' + product_id +'.IMG'  
    except:
        print("Image link does not exist")
    
    r = requests.get(img_url)
    
    with open(filepath + product_id + '.img','wb') as f:
       f.write(r.content)


def download_image_list(img_csv):
    img_list = pd.read_csv(img_csv)
    product_ids = img_list[['PRODUCT_ID']].to_numpy()
    for pid in product_ids:
        download_image(pid)
        

       



       

       
    
    
    





