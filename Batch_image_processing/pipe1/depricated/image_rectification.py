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
from dcnn_image_pipeline.scripts.rectification_tools_all_sites import * # Created by RL

def image_rectification(PanelDate, image, out_dir):
    # create specific output directory within out_dir
    ct = dt.datetime.now() # current time
    year = str(ct.year)
    month = str(ct.month)
    day = str(ct.day)

    out_dir = (out_dir + year + "_" + month + "_" + day)

    if not os.path.isfile(out_dir):
        os.mkdir(out_dir)
    
    files_jpg = [os.path.basename(image)]
    rootfolder = os.path.dirname(image) + os.sep
    
    batch_rectify_hi_res(PanelDate, files_jpg, rootfolder, out_dir)

    