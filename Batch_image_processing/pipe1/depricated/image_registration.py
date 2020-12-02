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

# Function from local directory for testing
from dcnn_image_pipeline.scripts.registration_tools import *    # Created by RL

# run this to complete registration
# method can be 1, 2, or 3

def image_registration(ref_img_path, image, in_dir, out_dir, method = 1):
    unable_to_register = []

    if method == 1:
        output = register_ECC(ref_img_path, image, in_dir, out_dir, unable_to_register)
        ''' note that this method can utilize homography or translation. Homography seems to be more robust so it is the default for this function
        you can change to translation by adding warp_mode = cv2.MOTION_TRANSLATION
        also, if the image undistortion created really big black borders, sometimes this method fails, therefore there is an alternative function which
        crops in from the edges of the image before doing registration, its called
        register_ECC_Crop(ref_img_path,image, in_dir, out_dir,unable_to_register,warp_mode = cv2.MOTION_TRANSLATION,crop = 250)
        '''
    return [unable_to_register, output]
## method 2:
# for image in image_list:
#   register_dft(ref_img_path, image,in_dir, out_dir,unable_to_register)

## method 3:
# for image in image_list:
#   register_2dfft(ref_img_path, image,in_dir, out_dir,unable_to_register)