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
from scripts.rectification_tools import * # Created by RL
from scripts.registration_tools import * # Created by RL

class Pipeline_one():

    def __init__(self, out_path):
        """Initialize pipeline_one class"""
        self.out_path = out_path # path for all output folders and files


    def remove_distortion(self, SiteDistortion, image_file):
        """Remove distortion from images"""
        startTime = dt.datetime.now()
        ct = dt.datetime.now()
        year = str(ct.year)
        month = str(ct.month)
        day = str(ct.day)

        A_Mat = SiteDistortion.A_Mat
        A = SiteDistortion.A_MatT
        distCoeff = SiteDistortion.distCoeff
        calibration = SiteDistortion.cal_year
        site = SiteDistortion.sitename
        f = image_file

        (filename, ext) = os.path.splitext(f)
        filename = os.path.basename(filename)

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
        out_dir = self.out_path + os.sep + 'undistorted' + os.sep
        if not os.path.isdir(out_dir):
            os.mkdir(out_dir)
            print('Created_directory:', out_dir)


        filename = out_dir + os.sep + filename + '_und' + ext
        print(f'file {filename} saved')
        cv2.imwrite(filename, dst)
        print(f'number of files in out_dir: {len(os.listdir(out_dir))}')
        now = dt.datetime.now() - startTime
        print(filename, 'coverted', now)

        end = dt.datetime.now()
        elapsed_time = end - ct
        print("Processing Finished, time elsapsed:", elapsed_time)


        return([filename, dst])

    def image_registration(self, ref_img_path, image, in_dir, method = 1):
        """Align images with a reference image."""
        unable_to_register = []

        out_dir = self.out_path + os.sep + 'registered' + os.sep
        print(out_dir)
        if not os.path.isdir(out_dir):
            os.mkdir(out_dir)

        if method == 1:
            registered_image_full_path = register_ECC(ref_img_path, image, in_dir, out_dir, unable_to_register, numiter = 75)
            ''' note that this method can utilize homography or translation. Homography seems to be more robust so it is the default for this function
            you can change to translation by adding warp_mode = cv2.MOTION_TRANSLATION
            also, if the image undistortion created really big black borders, sometimes this method fails, therefore there is an alternative function which
            crops in from the edges of the image before doing registration, its called
            register_ECC_Crop(ref_img_path,image, in_dir, out_dir,unable_to_register,warp_mode = cv2.MOTION_TRANSLATION,crop = 250)
            '''

        # TODO Have not been tested in the new framework MKF
        ## method 2:
        # for image in image_list:
        if method == 2:
            registered_image_full_path = register_dft(ref_img_path, image, in_dir, out_dir, unable_to_register)

        ## method 3:
        # for image in image_list:
        if method == 3:
            registered_image_full_path = register_2dfft(ref_img_path, image, in_dir, out_dir, unable_to_register)

        return [unable_to_register, registered_image_full_path]


    def image_rectification(self, PanelDate, image_full_path):
        """Rectify image"""
        # create specific output directory within out_dir
        ct = dt.datetime.now() # current time
        year = str(ct.year)
        month = str(ct.month)
        day = str(ct.day)

        out_dir = self.out_path + os.sep + 'rectified' + os.sep
        if not os.path.isdir(out_dir):
            os.mkdir(out_dir)


        files_jpg = [os.path.basename(image_full_path)]
        rootfolder = os.path.dirname(image_full_path) + os.sep

        try:
            batch_rectify_hi_res(PanelDate, files_jpg, rootfolder, out_dir)
        except FileExistsError:
            print('File already exists.')

    def run_pipeline_one(self, image_file, ref_img_path, SiteDistortion, PanelDate):
        """Run pipeline one"""
        # Test Pipeline one class
        #pipe1 = pi1(out_path = 'images_mike/')

        # Run distortion
        print("STEP 1: REMOVING_DISTORTION.....................")
        testp1_2 = self.remove_distortion(SiteDistortion = SiteDistortion, image_file = image_file)

        # next we specify the location of images to be registered
        in_dir = self.out_path + os.sep + 'undistorted' + os.sep
        # Run registration


        image = os.path.basename(testp1_2[0])
        #print('image',image)
        print("STEP 2: REGISTERING IMAGE.....................")
        testp1_3 = self.image_registration(ref_img_path = ref_img_path, image = image, in_dir = in_dir, method = 1)

        # Run rectified
        out_dir = self.out_path + os.sep + 'rectified' + os.sep
        if os.path.isfile(testp1_3[1]):
            print("STEP 3: RECTIFYING IMAGE.....................")
            testp1_4 = self.image_rectification(PanelDate = PanelDate,  image_full_path = testp1_3[1])
        else:
            print(f'No image {testp1_3[1]} to rectify')
