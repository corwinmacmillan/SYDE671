#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# This script extends planetaryimage for loading EDR images

# It:
# 1) fixes the decoding bug when loading EDR images

# 2) adds support for decompressing the 8-bit values to 12-bit values (cast as 16-bit floats which numpy can handle)

# 3) checks md5 sums for file integrity

# 4) improves image updating and saving capability


import os
import hashlib

import numpy as np

import pvl
from planetaryimage import PDS3Image
import decompand



class PDS3ImageEDR(PDS3Image):
    """EDR image class, extends PDS3Image class by fixing EDR decoding bug, adding support for decompressing
    8-bit values to 12-bit instrument DN values and checking md5 file integrity.
    """
    
    # this fixes the decoding bug by overriding the class SAMPLE_TYPE and DTYPES attributes
    # i.e. the EDR image (self.data) is loaded correctly as 8-bit unsigned integers
    
    SAMPLE_TYPES = {
        'LSB_INTEGER': '<u',# nb PSD3Image figures out these are 8-bit using int(self.label['IMAGE']['SAMPLE_BITS'] / 8)
    }

    DTYPES = {
        '<u': 'LSB_INTEGER',
        '<f': 'PC_REAL',# allows images to be updated to floating points
    }
    
    
    # override the class initialisation to add md5 check and decompanding
    def __init__(self, *args, **kwargs):
        """
        Create an Image object.
        """
        super().__init__(*args, **kwargs)
        
        # md5 check
        assert hashlib.md5(self.data.tobytes()).hexdigest() == self.label["IMAGE"]["MD5_CHECKSUM"]
        
        # # product type check
        # assert self.label["PRODUCT_TYPE"] == "EDR"
        # assert self.label["PRODUCT_ID"].endswith("E")
        
        # get image
        image = self.data.squeeze()
        
        # The code below backs off the LROC companding tables, converting the
        # 8-bit unsigned integers to the original 12-bit unsigned integers
        # recorded on the instrument, cast as 16-bit unsigned integers 
        # (because numpy does not support 12-bit integers).
        table = self.label["LRO:COMPAND_CODE"]# integer table number
        assert self.label["LRO:BTERM"] == decompand._bterms[table]# check terms are as expected
        assert self.label["LRO:XTERM"] == decompand._xterms[table]
        assert self.label["LRO:MTERM"] == decompand._mterms[table]
        self._image = decompand.decompand(image.astype('uint8'), table)# decompanded image
        
        # get image identifier (using our naming scheme)
        self.PRODUCT_ID = self.label["PRODUCT_ID"].replace("E","C")
        assert self.PRODUCT_ID in os.path.basename(self.filename)
        
        
    # override image property (make it static)
    @property
    def image(self):
        return self._image
    
    
    # new method for updating image array in place
    def update_image(self, im):
        "Update the image array with some new data"
        
        if im.shape != self.image.shape: raise Exception("ERROR: input image is not the same shape as current image! (%s) (%s)"%(im.shape, self.image.shape))
        
        # update underlying data, force to be LSB
        self.data = np.expand_dims(im.copy(), 0).astype("<%s%i"%(im.dtype.kind, im.dtype.itemsize))# add band dimension
        self._image = self.data.squeeze()
        
        # update label
        self.label['IMAGE']['LINES'] = self.data.shape[1]
        self.label['IMAGE']['LINE_SAMPLES'] = self.data.shape[2]
        self.label["IMAGE"]["MD5_CHECKSUM"] = hashlib.md5(self.data.tobytes()).hexdigest()
        self.label['IMAGE']['SAMPLE_BITS'] = self.data.itemsize * 8
        self.label['IMAGE']['SAMPLE_TYPE'] = self.DTYPES["<" + self.dtype.kind]
        
        
    # override save: remove overwriting, instantiate PDSLabelEncoder and use encoder=encoder argument, clean up padding
    def save(self, filepath):
        """Save PDS3Image object as PDS3 file.
        """
        
        # TODO: update FILE_RECORDS? (might be ok because we set LINES, LINE_SAMPLES above..)
        
        encoder = PDSLabelEncoderEDR()
        
        # get image pointer and length of label in bytes
        LABEL_RECORDS = self.label['^IMAGE'] = 0
        while self.label['^IMAGE'] != LABEL_RECORDS+1:
            
            # update image pointer
            self.label['^IMAGE'] = LABEL_RECORDS+1
            
            # get length of label in bytes
            label_nbytes = len(pvl.dumps(self.label, encoder=encoder).encode("ascii"))
            LABEL_RECORDS = int(np.ceil(label_nbytes / self.label['RECORD_BYTES']))
            
            # update label records
            self.label["LABEL_RECORDS"] = LABEL_RECORDS
            
        # get padding required to make label fixed record length
        diff =  LABEL_RECORDS*self.label['RECORD_BYTES'] - label_nbytes
        
        # write label
        pvl.dump(self.label, filepath, encoder=encoder)
        
        with open(filepath, 'a') as stream:

            # write padding
            for i in range(0, diff):
                stream.write(" ")
            
            # write array as bytes
            self.data.tofile(stream)



class PDSLabelEncoderEDR(pvl.encoder.PDSLabelEncoder):
    """EDR PDS label encoder class, overrides time formatting (times without 'Z' character added)
    and 30 character limit for keys"""
    
    # override for skipping value error when keys are larger than 30
    def encode_assignment(self, key, value, level=0, key_len=None):
        
        if key_len is None:
            key_len = len(key)

        if len(key) > 30:
            pass

        ident = ''
        if((key.startswith('^') and self.is_assignment_statement(key[1:]))
           or self.is_assignment_statement(key)):
            ident = key.upper()
        else:
            raise ValueError(f'The keyword "{key}" is not a valid ODL '
                             'Identifier.')

        s = ''
        s += '{} = '.format(ident.ljust(key_len))
        s += self.encode_value(value)

        if self.end_delimiter:
            s += self.grammar.delimiters[0]

        return self.format(s, level)
    
    # override for removing "Z" in datetime strings
    def encode_time(self, value):# this is a static method in PVLEncoder
        return pvl.encoder.PVLEncoder.encode_time(value)
    
    
    
path = r'C:\Users\jon25\OneDrive - University of Waterloo\SYDE 671\SYDE671\Dark Summed\M107758599LC.IMG'   
    
    
if __name__ == "__main__":
    
    import matplotlib.pyplot as plt
    
    
    ## test reading
    
    
    I = PDS3Image
    print(I.SAMPLE_TYPES)
    
    I = PDS3Image.open(path)
    I.data = I.data.astype('uint8')
    print(I.data.dtype)
    print(I.data.min(), I.data.max())
    print(I.image.dtype)
    print(I.image.min(), I.image.max())
    
    # plot I.image
    plt.figure(figsize=(20,3))
    plt.imshow(I.image.astype(float), cmap="gray", vmin=20, vmax=60)
    plt.colorbar()
    plt.figure(figsize=(20,3))
    for i in range(5): plt.plot(I.image[i,:])
    
    
    print()
    I = PDS3ImageEDR
    print(I.SAMPLE_TYPES)
    
    I = PDS3ImageEDR.open(path)
    # I.data = I.data.astype('uint8')
    print(I.data.dtype)
    print(I.data.min(), I.data.max())
    print(I.image.dtype)
    print(I.image.min(), I.image.max())
    
    # plot I.image
    plt.figure(figsize=(20,3))
    plt.imshow(I.image.astype(float), cmap="gray", vmin=20, vmax=60)
    plt.colorbar()
    plt.figure(figsize=(20,3))
    for i in range(5): plt.plot(I.image[i,:])
    
    plt.show()
    
    
    ## test saving
    
    
    # straight save
    I = PDS3ImageEDR.open(path)

    plt.figure(figsize=(20,3))
    plt.imshow(I.image, cmap="gray", vmin=20, vmax=60)
    plt.colorbar()
    
    #I.data = I.data.astype(np.float64)
    I.save("M107724658LC.IMG")
    I = PDS3ImageEDR.open(path)
    
    print(I.label)
    print(I.data.dtype)
    print(I.data.shape)
    
    plt.figure(figsize=(20,3))
    plt.imshow(I.image, cmap="gray", vmin=20, vmax=60)
    plt.colorbar()
    
    # update and save
    I = PDS3ImageEDR.open(path)
    a1 = I.image.copy()
    
    plt.figure(figsize=(20,3))
    plt.imshow(I.image, cmap="gray", vmin=20, vmax=60)
    plt.colorbar()
    
    I.update_image(I.image.astype(np.float32))
    I.save("test")
    I = PDS3Image.open("test")
    a2 = I.image.copy()
    
    print(I.label)
    print(I.data.dtype)
    print(I.data.shape)
    print(np.allclose(a1.astype(int), a2.astype(int)))
    
    plt.figure(figsize=(20,3))
    plt.imshow(I.image, cmap="gray", vmin=20, vmax=60)
    plt.colorbar()