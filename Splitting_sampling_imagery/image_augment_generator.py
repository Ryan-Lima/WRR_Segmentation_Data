import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image, ImageFilter
import random
import cv2
from random import shuffle
from glob import glob
from imageio import imread, imwrite
import albumentations as A

batch_size = 3
size = (128,128)

image_files = #list of full image paths
label_files = #list of fill label paths

def image_generator_augment(image_files, label_files ,batch_size, sz):
  while True:
    #extract a random batch of image files
    batch_index = np.random.choice(len(image_files), size = batch_size, replace = False)

    #variables for collecting batches of inputs and outputs
    batch_x = []
    batch_y = []
    transform = A.Compose([
        A.RandomRotate90(),
        A.Flip(),
        A.Transpose(),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
        A.OneOf([
            A.OpticalDistortion(p=0.3),
            A.GridDistortion(p=.1),
            A.IAAPiecewiseAffine(p=0.3),
        ], p=0.2),
        A.OneOf([
            A.CLAHE(clip_limit=2),
            A.RandomBrightnessContrast()], p = 0.3)])

    for i in batch_index:

        #get the masks. Note that masks are png files
        mask = Image.open(glob(label_files[i])[0])
        mask = np.array(mask.resize(sz))
        raw = Image.open(image_files[i])
        raw = raw.resize(sz)
        raw = np.array(raw)
        augmented = transform(image = raw, mask = mask)
        raw = augmented['image']
        mask = augmented['mask']
        mask[mask == 0 ] = 0
        mask[mask > 0] = 1

        batch_y.append(mask)
        #check the number of channels because some of the images are RGBA or GRAY
        if len(raw.shape) == 2:
          raw = np.stack((raw,)*3, axis=-1)

        else:
          raw = raw[:,:,0:3]

        #raw = ((raw - np.mean(raw))/np.std(raw))#.astype('uint8')

        batch_x.append(raw/255.)

    #preprocess a batch of images and masks
    batch_x = np.array(batch_x)#/255.
    batch_y = np.array(batch_y)
    batch_y = np.expand_dims(batch_y,3)#/255.
    #visualize(batch_x[0], batch_y[0].squeeze())

    yield (batch_x, batch_y)


def visualize(image, mask, original_image=None, original_mask=None):
    fontsize = 18

    if original_image is None and original_mask is None:
        f, ax = plt.subplots(2, 1, figsize=(8, 8))

        ax[0].imshow(image)
        ax[1].imshow(mask)
    else:
        f, ax = plt.subplots(2, 2, figsize=(8, 8))

        ax[0, 0].imshow(original_image)
        ax[0, 0].set_title('Original image', fontsize=fontsize)

        ax[1, 0].imshow(original_mask)
        ax[1, 0].set_title('Original mask', fontsize=fontsize)

        ax[0, 1].imshow(image)
        ax[0, 1].set_title('Transformed image', fontsize=fontsize)

        ax[1, 1].imshow(mask)
        ax[1, 1].set_title('Transformed mask', fontsize=fontsize)

generator = image_generator_augment(image_files, label_files ,batch_size, sz)
X, Y = next(generator)
visualize(X[0],Y[0].squeeze())
visualize(X[1],Y[1].squeeze())
visualize(X[2],Y[2].squeeze())
