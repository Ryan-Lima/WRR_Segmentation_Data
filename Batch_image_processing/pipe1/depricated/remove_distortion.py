#!/usr/bin/env python3

# -*- coding: utf-8 -*-

import os
import scipy as sp
import glob
import numpy as np
#from imageio import imread
#import imreg_dft as ird
import matplotlib.pyplot as plt
import cv2
import datetime as dt

from PIL import Image
#!pip install imreg_dft # the ! is for google colab
import imreg_dft as ird
#from datetime import *
import time
import numpy.ma as ma
import imageio
from skimage import img_as_float
from scipy.interpolate import griddata

##############################################################################
#
# NOTE This will process ONE file at a time. The calling code should take care
# of applying this function to more than one image.
#
# TODO the function needs to get the name from the get_images() value
#      this will involve splitting the text based on os.sep
#
# Inputs
#  A_Mat
#  files_jpg
#  
#
##############################################################################

def remove_distortion(A_Mat, distCoeff, calibration, site, image_file):
    startTime = dt.datetime.now()
    ct = dt.datetime.now()
    year = str(ct.year)
    month = str(ct.month)
    day = str(ct.day)

    A = A_Mat.T
    f = image_file
        
    (filename, ext) = os.path.splitext(f)
    print(f"undistorting {filename} using {calibration} from {site}")
    #img = working_directory + os.sep + f # get_images() 
    img = f
    img = cv2.imread(img) # read image in
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    newcameramatx, roi = cv2.getOptimalNewCameraMatrix(A, distCoeff, (w, h), 1, (w, h))
    dst = cv2.undistort(img, A, distCoeff, None, newcameramatx)
    dst = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    
    # if out_dir != None:
    #    cv2.imwrite(os.path.join(out_dir, filename + "_und" + ext), dst)
    
    now = dt.datetime.now() - startTime
    print(filename, 'coverted', now)
    
    end = dt.datetime.now()
    elapsed_time = end - ct
    print("Processing Finished, time elsapsed:", elapsed_time)
    #print(len(files_jpg), "file converted")
    
    return(dst)