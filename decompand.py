#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# this script contains functions and conversion tables for companding and decompanding the NAC EDR images

# from LROC:
# In flight, the 12-bit raw DN values are companded to 8-bit values sent back to Earth and written in the EDRs. 
# This companding uses one of six on-board schemes shown in Fig. 16. To enable fast pixel readout, each bit-compression 
# scheme consists of a piecewise linear function composed of up to 5 segments. Most images use companding table 0. 
# Table 1 (8-bit equals 12-bit) is used for darks. Table 3 samples low DN more densely, and it is used frequently 
# for dim images. On the ground, the first step in the image processing pipeline is to take the 8-bit values in 
# the EDRs and calculate a floating point value that approximates the original 12-bit integer value. The approximation
# calculated is the unweighted mean of all the 12-bit raw DN values that would be companded the 8-bit value found 
# in the EDR.

# we convert using the companding psudeocode in LROCSIS.PDF and the decompanding script here (converted to python):
# https://github.com/USGS-Astrogeology/ISIS3/blob/dev/isis/src/lro/apps/lronac2isis/main.cpp


import numpy as np


# DEFINE COMPANDING TABLES (see LROCSIS.PDF)

_bterms = \
{0: [0, 8, 25, 59, 128, 0],
 1: [0, 0, 0, 0, 0, 0],
 2: [0, 0, 0, 0, 0, 0],
 3: [0, 16, 69, 103, 128, 0],
 4: [0, 0, 0, 65, 128, 0],
 5: [0, 0, 14, 65, 128, 0]}

# x-terms correspond to the 12-bit DN range (see appendix B)
_xterms = {
    0: [0, 32, 136, 543, 2207, 4095],
    1: [511, 0, 0, 0, 0, 4095],
    2: [0, 0, 0, 0, 4095, 4095],
    3: [0, 64, 424, 536, 800, 4095],
    4: [0, 0, 0, 1040, 2000, 4095],
    5: [0, 0, 112, 816, 2000, 4095]
 }

_mterms = \
{0: [0.5, 0.25, 0.125, 0.0625, 0.03125, 0],
 1: [0.5, 0.25, 0.125, 0.0625, 0.03125, 0],
 2: [0.5, 0.25, 0.125, 0.0625, 0.03125, 0],
 3: [0.5, 0.25, 0.125, 0.0625, 0.03125, 0],
 4: [0.5, 0.25, 0.125, 0.0625, 0.03125, 0],
 5: [0.5, 0.25, 0.125, 0.0625, 0.03125, 0]}

_MAX_INPUT_VALUE = 4095

def _get_decompand_maps():
    """"Pre-compute decompanding conversion maps for fast decompanding.
    This is a python implementation of this ISIS script:
    # https://github.com/USGS-Astrogeology/ISIS3/blob/dev/isis/src/lro/apps/lronac2isis/main.cpp
    """
    
    base = np.arange(256, dtype=np.uint8)
    decompand_maps = {}
    
    for table in range(6):
        
        decompand_maps[table] = np.zeros(256, dtype=np.uint16)
        
        # get companding table
        xterm = _xterms[table]# the y-intercept of the linear function
        bterm = _bterms[table]# the inflection point on the x-axis (12-bit axis)
        mterm = _mterms[table]# the gradient?
        
        for i,p in enumerate(base):
            
            # if pixin < xterm0, then it is in "segment 0"
            if p < xterm[0]:
                decompand_maps[table][i] = p
            
            # otherwise, it is in segments 1 to 5
            else:
                segment = 1
                while segment < len(xterm) and (p - bterm[segment - 1]) / mterm[segment - 1] >= xterm[segment]:
                    segment += 1
            
                # Compute the upper and lower bin values
                upper = (p + 1 - bterm[segment - 1]) / mterm[segment - 1] - 1
                lower = (p - bterm[segment - 1]) / mterm[segment - 1]
                
                # Check if the bin is on the upper boundary of the last segment
                if(upper > _MAX_INPUT_VALUE):
                    upper = _MAX_INPUT_VALUE
                elif segment < len(xterm) and upper >= xterm[segment]:
                    if int(bterm[segment] + mterm[segment]*upper) != p:
                        upper = xterm[segment] - 1
                    
                # Check if it is on the lower boundary of a segment
                if lower < xterm[segment-1]:
                    lower = xterm[segment-1]
                  
                # Output the middle bin value
                decompand_maps[table][i] = (upper + lower) / 2.0# note casts as integer value if array is uint
            
    return decompand_maps
    
decompand_maps = _get_decompand_maps()

def _get_compand_maps():
    """"Pre-compute companding conversion maps for fast companding.
    This is a python implementation of the psuedocode in LROCSIS.PDF.
    """
    
    base = np.arange(65536, dtype=np.uint16)
    compand_maps = {}
    
    for table in range(6):
        
        compand_maps[table] = np.zeros(65536, dtype=np.uint8)
    
        # get companding table
        xterm = _xterms[table]# the y-intercept of the linear function
        bterm = _bterms[table]# the inflection point on the x-axis (12-bit axis)
        
        for i,p in enumerate(base):
            
            if p < xterm[0]:
                pout = p%256
            elif p < xterm[1]: 
                pout = p/2+bterm[0]
            elif p < xterm[2]: 
                pout = p/4+bterm[1]
            elif p < xterm[3]: 
                pout = p/8+bterm[2]
            elif p < xterm[4]: 
                pout = p/16+bterm[3]
            elif p < xterm[5]:
                pout = p/32+bterm[4]
            else:
                pout = 0
            
            # Output the middle bin value
            compand_maps[table][i] = pout# note casts as integer value if array is uint
            
    return compand_maps

compand_maps = _get_compand_maps()


    
def decompand(image, table):
    "Decompand an 8-bit NAC image to 12-bits, cast as 16-bit integers"
    
    if image.dtype != np.uint8: raise Exception("ERROR: image is not np.uint8! (%s)"%(image.dtype))
    if image.ndim != 2: raise Exception("ERROR: image dimension is not 2! (%s)"%(image.shape,))
    if table not in np.arange(6): raise Exception("ERROR: table not in 0-5! (%s)"%(table))
    
    # copy is probably not needed, but do it for safety
    # https://scipy-cookbook.readthedocs.io/items/ViewsVsCopies.html#I-think-I-understand-what-a-view-is,-but-why-fancy-indexing-is-not-returning-a-view?
    image = decompand_maps[table][image].copy()# map
    assert image.dtype == np.uint16
    return image

def compand(image, table):
    "Compand a 12-bit NAC image, cast as 16-bit integers, to 8-bit"
    
    if image.dtype != np.uint16: raise Exception("ERROR: image is not np.uint16! (%s)"%(image.dtype))
    if image.ndim != 2: raise Exception("ERROR: image dimension is not 2! (%s)"%(image.shape,))
    if table not in np.arange(6): raise Exception("ERROR: table not in 0-5! (%s)"%(table))
    
    # copy is probably not needed, but do it for safety
    # https://scipy-cookbook.readthedocs.io/items/ViewsVsCopies.html#I-think-I-understand-what-a-view-is,-but-why-fancy-indexing-is-not-returning-a-view?
    image = compand_maps[table][image].copy()# map
    assert image.dtype == np.uint8
    return image

if __name__ == "__main__":
    
    import time
    import matplotlib.pyplot as plt
    
    
    # # plot companding / decompanding maps
    # plt.figure(figsize=(10,10))
    
    # base = np.arange(65536, dtype=np.uint16)
    # for table in range(6):
    #     plt.plot(base, compand_maps[table], lw=3, label=table)
    
    # base = np.arange(256, dtype=np.uint8)
    # for table in range(6):
    #     plt.plot(decompand_maps[table], base, color="k", label=table)
    
    # plt.legend()
    # plt.show()
    
    
    # test using realistic image size (250 MB)
    t = 3 # Companding scheme 3 (in paper)
    
    '''
    Import LROC image (16 bit)
    '''
    # np.random.seed(123)
    # image = np.random.randint(0, 65536, size=(52224, 5064), dtype=np.uint16)
    from planetaryimage import PDS3Image
    I = PDS3Image.open('SYDE671\Dark Normal\M107724658LC.IMG')
    image = I.image.astype(np.uint16)
    
    fig1 = plt.figure()
    plt.imshow(image, cmap='gray')
    
    

    '''
    Compand to 8 bit
    '''    
    start = time.time()
    image2 = compand(image, t)
    print("%.2f s"%(time.time()-start))
    
    '''
    Decompand back to 16 bit
    '''
    start = time.time()
    image3 = decompand(image2, t)
    print("%.2f s"%(time.time()-start))
    
    plt.figure()
    plt.scatter(image.flatten()[:1000], image2.flatten()[:1000], s=2)
    
    plt.figure()
    plt.scatter(image3.flatten()[:1000], image2.flatten()[:1000], s=2)
    
    '''
    OUTPUTS COMPANDING NOISE FIGURE
    '''
    noise = image3.flatten().astype(float)-image.flatten().astype(float)
    plt.figure()
    plt.scatter(np.arange(image3.size), image3.flatten().astype(float)-image.flatten().astype(float), s=1)
    
    plt.show()

    