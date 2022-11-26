#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# This script provides my own (adapted) ISIS nonlinearity and flatfield corrections

import itertools

import numpy as np
import pandas as pd
import scipy.interpolate

import planetaryimage


class MyIsis:
    "My class for carrying out (adapted) ISIS nonlinearity and flatfield correction"
    
    def __init__(self, isisdir="/data/users/bmoseley/psr-enhancement/6_benchmark/isis/"):
        "load ISIS calibration coefficents"
        
        modes = ["normal", "summed"]
        cameras = ["L", "R"]
        settings = list(itertools.product(modes, cameras))
        
        # 1. load raw offsets / flatfields / coeffs
        
        flatfields, offsets, coeffs = {}, {}, {}
        
        for mode, camera in settings:
            k = "%s_%s"%(mode, camera)
            tag = "_Summed" if mode == "summed" else ""
            
            flatfields[k] = \
            planetaryimage.CubeFile.open(
                isisdir+"NAC%s_Flatfield%s.0006.cub"%(camera, tag)).image
        
            offsets[k] = \
            planetaryimage.CubeFile.open(
                isisdir+"NAC%s_LinearizationOffsets%s.0006.cub"%(camera, tag)).image
            
            coeffs[k] = \
            np.array(pd.read_csv(
                isisdir+"NAC%s_LinearizationCoefficients.0007.txt"%(camera), header=None, sep='\s+'))
        
        # 2. get my versions of these
        
        myflatfields, myoffsets, mycoeffs = {},{},{}
        
        for k in flatfields:# these vary by mode and camera
            flatfield = flatfields[k].copy()
            flatfield[flatfield<0] = 1# set masked pixels to 1
            myflatfields[k] = flatfield
            
        for k in offsets:
            myoffsets[k] = np.mean(offsets[k][100:-100])# avoid masked values
        myoffsets["summed_L"] = myoffsets["normal_L"]# set to be the same across both modes
        myoffsets["summed_R"] = myoffsets["normal_R"]
        
        for k in coeffs:# these are already the same across both modes
            mycoeffs[k] = np.mean(coeffs[k][100:-100,:], axis=0)# avoid masked values
            mycoeffs[k][2] = (1/myoffsets[k]) - (mycoeffs[k][0]*(mycoeffs[k][1]**myoffsets[k]))# force curve through zero
            
        self.offsets = offsets
        self.flatfields = flatfields
        self.coeffs = coeffs

        self.myoffsets = myoffsets
        self.myflatfields = myflatfields
        self.mycoeffs = mycoeffs
        
        # 3. pre-define forward/ inverse functions for the nonlinearity correction
        
        nonlinearity_reverse = lambda k,im: (im+self.myoffsets[k]) - (1/((self.mycoeffs[k][0]*(self.mycoeffs[k][1]**(im+self.myoffsets[k]))) + self.mycoeffs[k][2]))# AS PER ISIS
        nonlinearity_forwards = {}
        for k in myoffsets:
            d = np.arange(-10, 2**12, 0.005)# use approximate numerical inversion for forward
            t = nonlinearity_reverse(k,d)
            forward = scipy.interpolate.interp1d(t, d, kind="linear", fill_value="extrapolate")# ok to extrapolate slightly because both curves go through zero and psuedo-linear for high DN
            nonlinearity_forwards[k] = forward
        nonlinearity_forward = lambda k,im: nonlinearity_forwards[k](im)
        
        self._nonlinearity_reverse = nonlinearity_reverse
        self._nonlinearity_forward = nonlinearity_forward

    def _validate(self, image, im, df):
        "validate inputs to MyIsis functions"
        
        mode = "normal" if df.loc[image]["LINE_SAMPLES"]==5064 else "summed"
        camera = image[-2]
        k = "%s_%s"%(mode, camera)
        
        # check image array is right shape
        if im.ndim != 2:
            raise Exception("ERROR: 2D image expected, got %s"%(im.shape,))
        if im.shape[1] != df.LINE_SAMPLES[image]:
            raise Exception("ERROR: number of line samples in input array (%s) does not match index! (%s)"%(im.shape[1], df.loc[image]))
        
        return k
    
    def flatfield(self, im, k='summed_L', forward=False):
        "run my own flatfield correction on an image array im"
        
        # k = self._validate(image, im, df)
        
        # AS PER ISIS
        if forward:
            return im * self.myflatfields[k]# works via broadcasting
        else:
            return im / self.myflatfields[k]

    def nonlinearity(self, im, k='summed_R', forward=False):
        "run my own (constant) nonlinearity correction on an image array im"
        
        # k = self._validate(image, im, df)
        
        if forward:
            return self._nonlinearity_forward(k,im)# both work via broadcasting
        else:
            return self._nonlinearity_reverse(k,im)
    
    def __repr__(self):
        "fancy print"
        s = ""
        for d in [self.myflatfields, self.myoffsets, self.mycoeffs]:
            for k in d:
                s+= "%s %s\n"%(k, d[k])
        return s.rstrip("\n")
    
if __name__ == "__main__":
    
    import matplotlib.pyplot as plt
    
    I = MyIsis(isisdir="/home/panlab/anaconda3/envs/isis/data/lro/calibration/")
        
    
    ## check values
    
    print(I)
    
    for k in I.flatfields:
        flatfield = I.flatfields[k].copy()
        flatfield[flatfield<0] = 0
        plt.plot(flatfield, label=k)
        plt.plot(I.myflatfields[k], label="my %s"%(k))
        plt.legend()
        plt.show()
    
    for k in I.offsets:
        plt.plot(I.offsets[k], label=k)
        plt.hlines(I.myoffsets[k], 0, I.offsets[k].shape[0], label="my %s"%(k), zorder=10)
        plt.legend()
        plt.show()
    
    for k in I.coeffs:
        for i in range(I.coeffs[k].shape[1]):
            plt.plot(I.coeffs[k][:,i], label="%s_%i"%(k,i))
            plt.hlines(I.mycoeffs[k][i], 0, I.coeffs[k].shape[0], label="my %s_%i"%(k,i), zorder=10)
        plt.legend()
        plt.show()
        
    
    
    ## check corrections
    
    df = pd.DataFrame([[5064],[5064],[5064//2],[5064//2]],
                      index=["M1149842753LC","M1149842753RC","M1149842754LC","M1149842754RC"], 
                      columns=["LINE_SAMPLES"])
    
    # check non-linearity correction
    for image in df.index:
        
        im = np.linspace(-10,100,df.LINE_SAMPLES[image]).reshape((1,df.LINE_SAMPLES[image]))
        im = np.concatenate([im]*10, axis=0)
        
        im2 = I.nonlinearity(image, im, df, forward=False)
        im3 = I.nonlinearity(image, im, df, forward=True)
        
        plt.plot(im[0], im2[0], linewidth=3, label="inverse")
        plt.plot(im2[0], im[0], linewidth=3)
        plt.plot(im[0], im[0])
        plt.plot(im[0], im3[0], label="forward")
        plt.plot(im3[0], im[0])
    plt.legend()
    plt.show()
    print(im.shape, im2.shape, im3.shape)
    
    # check flatfield correction
    for image in df.index:
        im = np.ones(df.LINE_SAMPLES[image]).reshape((1,df.LINE_SAMPLES[image]))
        im = np.concatenate([im]*10, axis=0)
        
        imi = I.flatfield(image, im, df, forward=False)
        imf = I.flatfield(image, im, df, forward=True)
        
        plt.plot(im[0])
        plt.plot(imi[0])
        plt.plot(imf[0])
        plt.show()
    print(im.shape, imi.shape, imf.shape)
    