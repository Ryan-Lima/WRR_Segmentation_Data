# function to check images for Glare

import glob
import os
import datetime as dt
from datetime import time
import cv2
import matplotlib.pyplot as plt
from PIL import Image

def is_time_between(begin_time, end_time, check_time=None):
    # If check time is not given, default to current UTC time
    check_time = check_time or dt.datetime.utcnow().time()
    if begin_time < end_time:
        return check_time >= begin_time and check_time <= end_time
    else: # crosses midnight
        return check_time >= begin_time or check_time <= end_time

def create_img_list_chk_glare(img_path_list, Start_time = '1500', End_time = '1700'):
  '''
  This function allows for checking images
  which were taken at a certain time of day. For some sites the sun shines into
  the camera lens at certain times of the year making the photos unusable. Rather
  than waste computing time processing bad images, it allows us to check the images
  prior to DCNN prediction. For RC0307Rf for example, bad images are often taken
  between 1500 and 1700.
  INPUT - This function requires:
    img_list - list of potential images
    Start_time - start time for image check as 4 digit string 24 hour ex. '1500'
    End_time - end time for image check as 4 digit string 24 hour ex. '1700'
  OUTPUT - This function returns:
    img_path_list - a list of images with the full path (needed for loading an image)
    img_list - as list of just the image basenames
    filenames_list - a list of the filenames basenames without extensions
  '''

  good_images = []
  bad_images = []
  print(f'Manual Checking of images between {Start_time} and {End_time}')
  time1 = dt.datetime.strptime(Start_time,"%H%M")
  time2 = dt.datetime.strptime(End_time, "%H%M")
  imgs_path_list = []
  filenames_list = []
  img_dir , _ = os.path.split(img_path_list[0])
  for file in img_path_list:
    path, basename = os.path.split(file)
    filename, ext = os.path.splitext(basename)
    split_name_list = filename.split('_')
    #print(split_name_list)
    image_time = dt.datetime.strptime(split_name_list[2],"%H%M")
    if is_time_between(time1,time2, image_time):

      badim = Image.open(path + os.sep + basename)
      plt.imshow(badim)
      plt.pause(2)
      try_again = True
      while try_again == True:
        answer = input(f'Use image {basename}?.. Enter Y/N:')
        if answer == "Y":
          good_images.append(basename)
          print(f'{basename} added to image list')
          try_again = False
        elif answer == "N":
          bad_images.append(basename)
          print(f'{basename} excluded from image list')
          try_again = False
        else:
          print('Answer not valid')
    else:
      good_images.append(basename)
  print(f'The following images have been excluded: {bad_images}')
  img_list = good_images
  for f in img_list:
    image_path_full = img_dir + os.sep + f
    imgs_path_list.append(image_path_full)
    filename, ext = os.path.splitext(f)
    filenames_list.append(filename)
  return imgs_path_list, img_list, filenames_list
