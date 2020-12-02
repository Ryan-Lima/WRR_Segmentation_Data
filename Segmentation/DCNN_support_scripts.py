''' support scripts for UNet  binary image segmentation with Python,
Keras, and Tensorflow

'''

#imports
#! pip install pydensecrf
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
from sklearn.model_selection import train_test_split
from skimage.morphology import binary_erosion, binary_dilation, square
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization
from tensorflow.keras.layers import Concatenate, Conv2DTranspose, Flatten, Activation, Add
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Conv2D, Conv2DTranspose
from tensorflow.keras.constraints import max_norm
from tensorflow.keras import backend as K
from skimage.transform import resize
from statistics import mean
import sklearn.metrics
import datetime
import json, codecs
import albumentations as A


class model_dictionary(dict):
  def __init__(self):
    self = dict()

  def add(self, key, value):
    self[key] = value

callbacks = tf.keras.callbacks
backend = tf.keras.backend

class PlotLearning(tf.keras.callbacks.Callback):

    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        self.acc = []
        self.val_acc = []
        #self.fig = plt.figure()
        self.logs = []
    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.acc.append(logs.get('dice_coef'))
        self.val_acc.append(logs.get('val_dice_coef'))
        self.i += 1
        print('i=',self.i,'loss=',logs.get('loss'),'val_loss=',logs.get('val_loss'),'dice_coef=',logs.get('dice_coef'),'val_dice_coef=',logs.get('val_dice_coef'))


        #choose a random test image and preprocess
        path = np.random.choice(VAL_images)
        raw = Image.open(path) # open image
        raw = np.array(raw.resize(sz[:2]))/255.
        raw = raw[:,:,0:3]
        #predict the mask
        pred = model.predict(np.expand_dims(raw, 0))
        #mask post-processing
        msk  = pred.squeeze()
        msk = np.stack((msk,)*3, axis=-1)
        msk = msk*255
        msk[msk >= 0.5] = 1
        msk[msk < 0.5] = 0

        #show the mask and the segmented image
        combined = np.concatenate([raw, msk, raw* msk], axis = 1)
        plt.axis('off')
        plt.imshow(combined)
        plt.show(block = False)
        plt.close('all')

class LearningRateScheduler(callbacks.Callback):
    def __init__(self,
                 schedule,
                 learning_rate=None,
                 steps_per_epoch=None,
                 verbose=0):
        super(LearningRateScheduler, self).__init__()
        self.learning_rate = learning_rate
        self.schedule = schedule
        self.verbose = verbose
        self.warmup_epochs = 0
        self.warmup_steps = 0
        self.global_batch = 0

    def on_train_batch_begin(self, batch, logs=None):
        self.global_batch += 1
        if self.global_batch < self.warmup_steps:
            if not hasattr(self.model.optimizer, 'lr'):
                raise ValueError('Optimizer must have a "lr" attribute.')
            lr = self.learning_rate * self.global_batch / self.warmup_steps
            backend.set_value(self.model.optimizer.lr, lr)
            if self.verbose > 0:
                print('\nBatch %05d: LearningRateScheduler warming up learning '
                      'rate to %s.' % (self.global_batch, lr))

    def on_epoch_begin(self, epoch, logs=None):
        if not hasattr(self.model.optimizer, 'lr'):
            raise ValueError('Optimizer must have a "lr" attribute.')
        lr = float(backend.get_value(self.model.optimizer.lr))

        if epoch >= self.warmup_epochs:
            try:  # new API
                lr = self.schedule(epoch - self.warmup_epochs, lr)
            except TypeError:  # old API
                lr = self.schedule(epoch - self.warmup_epochs)
            if not isinstance(lr, (float, np.float32, np.float64)):
                raise ValueError('The output of the "schedule" function '
                                 'should be float.')
            backend.set_value(self.model.optimizer.lr, lr)

            if self.verbose > 0:
                print('\nEpoch %05d: LearningRateScheduler reducing learning '
                      'rate to %s.' % (epoch + 1, lr))

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs['lr'] = backend.get_value(self.model.optimizer.lr)

# cosinse function implementation for cyclical learning rate between two specified bounds 'min_lr' and 'max_lr'

def cosine_ratedecay(max_epochs, max_lr, min_lr=1e-6):
    """
    cosine scheduler.
    :param max_epochs: max epochs
    :param max_lr: max lr
    :param min_lr: min lr
    :return: current lr
    """
    max_epochs = max_epochs ##- 5 if warmup else max_epochs

    def ratedecay(epoch):
        lrate = min_lr + (max_lr - min_lr) * (
                1 + np.cos(np.pi*2 * epoch / max_epochs)) / 2

        return lrate

    return ratedecay

#lr_ratedecay = cosine_ratedecay(epochs,max_lr) # needs to be in the script

class TimeHistory(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, epoch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, epoch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)

time_callback = TimeHistory()

# build callbacks
def build_callbacks(filepath, lr_ratedecay, lr, steps_per_epoch,time_callback):

    # set checkpoint file
    model_checkpoint = ModelCheckpoint(filepath, monitor='val_loss',
                                   verbose=2, save_best_only=True, mode='min',
                                   save_weights_only = True)

    # learning rate scheduler setting
    learning_rate_scheduler = LearningRateScheduler(lr_ratedecay, lr, steps_per_epoch,
                                                verbose=1)

    callbacks = [model_checkpoint, learning_rate_scheduler, PlotLearning()]

    return callbacks

'''
*********************************************MODELS************************************************
'''
sz = (128,128) # tuple of image dimensions


'''Unet model 1: this is a model Dan buscombe put together.
    > inputs
    > 6 steps Encoder path (contracting path) - each with:
        > Conv2D layer x 2
        > MaxPooling2D x 1

    > a "bottle neck?" not sure what to call it
    > 5 steps decoding (expanding path) each with:
        > conv2 layer x 2
        > Conv2DTranspose x 1
        > Concatenate x 1
    > 1 step prediction
        > Conv2D layer x 2
    > outputs
'''

def unet_1(sz):
    sz=sz + (3,)
    s = Input(sz)
    #Contraction path
    c1 = Conv2D(8,3, activation='relu',padding = 'same') (s) #conv2d = 128, 128, 8
    c1 = Conv2D(8,3, activation='relu',padding = 'same') (c1) #conv2d_1 = 128, 128, 8
    p1 = MaxPooling2D()(c1)

    c2 = Conv2D(16,3, activation='relu',padding = 'same') (p1) #conv2d_2 = 64, 64, 16
    c2 = Conv2D(16,3, activation='relu',padding = 'same') (c2) #conv2d_3 = 64, 64, 16
    p2 = MaxPooling2D()(c2)

    c3 = Conv2D(32,3, activation='relu',padding = 'same') (p2) #conv2d_4 = 32, 32, 32
    c3 = Conv2D(32,3, activation='relu',padding = 'same') (c3) #conv2d_5 = 32, 32, 32
    p3 = MaxPooling2D()(c3)

    c4 = Conv2D(64,3, activation='relu',padding = 'same') (p3) #conv2d_6 = 16, 16, 64
    c4 = Conv2D(64,3, activation='relu',padding = 'same') (c4) #conv2d_7 = 16, 16, 64
    p4 = MaxPooling2D()(c4)

    c5 = Conv2D(128,3, activation='relu',padding = 'same') (p4) #conv2d_8 = 8, 8, 128
    c5 = Conv2D(128,3, activation='relu',padding = 'same') (c5) #conv2d_9 = 8, 8, 128
    p5 = MaxPooling2D()(c5)

    c6 = Conv2D(256,3, activation='relu',padding = 'same') (p5) #conv2d_10 = 4, 4, 256
    c6 = Conv2D(256,3, activation='relu',padding = 'same') (c6) #conv2d_11 = 4, 4, 256
    p6 = MaxPooling2D()(c6)

    c7 = Conv2D(512,3, activation='relu',padding = 'same') (p6)  # conv2d_12 = 2, 2, 512
    c7 = Conv2D(512,3, activation='relu',padding = 'same') (c7) # conv2d_13 = 2, 2, 512
    c7 = Conv2DTranspose(64, 2,strides=(2, 2), padding='same') (c7) # conv2dTranspose = 4, 4, 64
    c7 = Concatenate()([c7,c6])  # concatenate [conv2dTranspose, conv2D_11] =  4, 4, 320

    u8 = Conv2D(256,3, activation='relu',padding = 'same') (c7) # conv2d_14 = 4, 4, 256
    u8 = Conv2D(256,3, activation='relu',padding = 'same') (u8) # conv2d_15 = 4, 4, 256
    u8 = Conv2DTranspose(32, 2,strides=(2, 2), padding='same') (u8) # convtdTranspose_1 = 8, 8, 32
    u8 = Concatenate() ([u8,c5]) # concatenate[conv2dTranpose_1, conv2D_9]

    u9 = Conv2D(128,3, activation='relu',padding = 'same') (u8) # conv2d_16 = 8, 8, 128
    u9 = Conv2D(128,3, activation='relu',padding = 'same') (u9) # conv2d_17 = 8, 8, 128
    u9 = Conv2DTranspose(16, 2,strides=(2, 2), padding='same') (u9) # conv2dTranpose_2 = 16, 16, 16
    u9 = Concatenate() ([u9,c4]) # concatenate[conv2dTranpose_2, conv2D_7] = 16, 16, 80

    u10 = Conv2D(64,3, activation='relu',padding = 'same') (u9)
    u10 = Conv2D(64 ,3, activation='relu',padding = 'same') (u10)
    u10 = Conv2DTranspose(8, 2,strides=(2, 2), padding='same') (u10)
    u10 = Concatenate() ([u10,c3]) # concatenate[conv2dTranpose_3, conv2D_5]

    u11 = Conv2D(32,3, activation='relu',padding = 'same') (u10)
    u11 = Conv2D(32,3, activation='relu',padding = 'same') (u11)
    u11 = Conv2DTranspose(4, 2,strides=(2, 2), padding='same') (u11)
    u11 = Concatenate() ([u11,c2])

    u12 = Conv2D(16,3, activation='relu',padding = 'same') (u11)
    u12 = Conv2D(16,3, activation='relu',padding = 'same') (u12)
    u12 = Conv2DTranspose(2, 2,strides=(2, 2), padding='same') (u12)
    u12 = Concatenate()([u12,c1])

    pr12 = Conv2D(16,3, activation='relu',padding = 'same') (u12)
    pr12 = Conv2D(16,3, activation='relu',padding = 'same') (pr12)
    outputs = Conv2D(1,1, activation = 'sigmoid') (pr12)
    model = Model(inputs=[s], outputs=[outputs])
    return model

''' Unet with dropouts. similar structure to Dans model but with dropouts

'''

def unet_2(sz):
    sz=sz + (3,)
    s = Input(sz)
    #Contraction path
    c1 = Conv2D(8,3, activation='relu',padding = 'same') (s) #conv2d = 128, 128, 8
    c1 = Dropout(0.1) (c1)
    c1 = Conv2D(8,3, activation='relu',padding = 'same') (c1) #conv2d_1 = 128, 128, 8
    p1 = MaxPooling2D()(c1)

    c2 = Conv2D(16,3, activation='relu',padding = 'same') (p1) #conv2d_2 = 64, 64, 16
    c2 = Dropout(0.1) (c2)
    c2 = Conv2D(16,3, activation='relu',padding = 'same') (c2) #conv2d_3 = 64, 64, 16
    p2 = MaxPooling2D()(c2)

    c3 = Conv2D(32,3, activation='relu',padding = 'same') (p2) #conv2d_4 = 32, 32, 32
    c3 = Dropout(0.1) (c3)
    c3 = Conv2D(32,3, activation='relu',padding = 'same') (c3) #conv2d_5 = 32, 32, 32
    p3 = MaxPooling2D()(c3)

    c4 = Conv2D(64,3, activation='relu',padding = 'same') (p3) #conv2d_6 = 16, 16, 64
    c4 = Dropout(0.2) (c4)
    c4 = Conv2D(64,3, activation='relu',padding = 'same') (c4) #conv2d_7 = 16, 16, 64
    p4 = MaxPooling2D()(c4)

    c5 = Conv2D(128,3, activation='relu',padding = 'same') (p4) #conv2d_8 = 8, 8, 128
    c5 = Dropout(0.2) (c5)
    c5 = Conv2D(128,3, activation='relu',padding = 'same') (c5) #conv2d_9 = 8, 8, 128
    p5 = MaxPooling2D()(c5)

    c6 = Conv2D(256,3, activation='relu',padding = 'same') (p5) #conv2d_10 = 4, 4, 256
    c6 = Dropout(0.2) (c6)
    c6 = Conv2D(256,3, activation='relu',padding = 'same') (c6) #conv2d_11 = 4, 4, 256
    p6 = MaxPooling2D()(c6)

    c7 = Conv2D(512,3, activation='relu',padding = 'same') (p6)  # conv2d_12 = 2, 2, 512
    c7 = Dropout(0.3) (c7)
    c7 = Conv2D(512,3, activation='relu',padding = 'same') (c7) # conv2d_13 = 2, 2, 512
    c7 = Conv2DTranspose(64, 2,strides=(2, 2), padding='same') (c7) # conv2dTranspose = 4, 4, 64
    c7 = Concatenate()([c7,c6])  # concatenate [conv2dTranspose, conv2D_11] =  4, 4, 320

    u8 = Conv2D(256,3, activation='relu',padding = 'same') (c7) # conv2d_14 = 4, 4, 256
    u8 = Dropout(0.2) (u8)
    u8 = Conv2D(256,3, activation='relu',padding = 'same') (u8) # conv2d_15 = 4, 4, 256
    u8 = Conv2DTranspose(32, 2,strides=(2, 2), padding='same') (u8) # convtdTranspose_1 = 8, 8, 32
    u8 = Concatenate() ([u8,c5]) # concatenate[conv2dTranpose_1, conv2D_9]

    u9 = Conv2D(128,3, activation='relu',padding = 'same') (u8) # conv2d_16 = 8, 8, 128
    u9 = Dropout(0.2) (u9)
    u9 = Conv2D(128,3, activation='relu',padding = 'same') (u9) # conv2d_17 = 8, 8, 128
    u9 = Conv2DTranspose(16, 2,strides=(2, 2), padding='same') (u9) # conv2dTranpose_2 = 16, 16, 16
    u9 = Concatenate() ([u9,c4]) # concatenate[conv2dTranpose_2, conv2D_7] = 16, 16, 80

    u10 = Conv2D(64,3, activation='relu',padding = 'same') (u9)
    u10 = Dropout(0.2) (u10)
    u10 = Conv2D(64 ,3, activation='relu',padding = 'same') (u10)
    u10 = Conv2DTranspose(8, 2,strides=(2, 2), padding='same') (u10)
    u10 = Concatenate() ([u10,c3]) # concatenate[conv2dTranpose_3, conv2D_5]

    u11 = Conv2D(32,3, activation='relu',padding = 'same') (u10)
    u11 = Dropout(0.2) (u11)
    u11 = Conv2D(32,3, activation='relu',padding = 'same') (u11)
    u11 = Conv2DTranspose(4, 2,strides=(2, 2), padding='same') (u11)
    u11 = Concatenate() ([u11,c2])

    u12 = Conv2D(16,3, activation='relu',padding = 'same') (u11)
    u12 = Dropout(0.2) (u12)
    u12 = Conv2D(16,3, activation='relu',padding = 'same') (u12)
    u12 = Conv2DTranspose(2, 2,strides=(2, 2), padding='same') (u12)
    u12 = Concatenate()([u12,c1])

    pr12 = Conv2D(16,3, activation='relu',padding = 'same') (u12)
    pr12 = Conv2D(16,3, activation='relu',padding = 'same') (pr12)
    outputs = Conv2D(1,1, activation = 'sigmoid') (pr12)
    model = Model(inputs=[s], outputs=[outputs])
    return model

'''
Residual Unet model. not altogether clear how this one works.
'''
def batchnorm_act(x):
    x = BatchNormalization()(x)
    return Activation("relu")(x)

# and this
def conv_block(x, filters, kernel_size=(3, 3), padding="same", strides=1):
    conv = batchnorm_act(x)
    return Conv2D(filters, kernel_size, padding=padding, strides=strides)(conv)

# and this
def bottleneck_block(x, filters, kernel_size=(3, 3), padding="same", strides=1):
    conv = Conv2D(filters, kernel_size, padding=padding, strides=strides)(x)
    conv = conv_block(conv, filters, kernel_size=kernel_size, padding=padding, strides=strides)

    bottleneck = Conv2D(filters, kernel_size=(1, 1), padding=padding, strides=strides)(x)
    bottleneck = batchnorm_act(bottleneck)

    return Add()([conv, bottleneck])

# what does it do?
def res_block(x, filters, kernel_size=(3, 3), padding="same", strides=1):
    res = conv_block(x, filters, kernel_size=kernel_size, padding=padding, strides=strides)
    res = conv_block(res, filters, kernel_size=kernel_size, padding=padding, strides=1)

    bottleneck = Conv2D(filters, kernel_size=(1, 1), padding=padding, strides=strides)(x)
    bottleneck = batchnorm_act(bottleneck)

    return Add()([bottleneck, res])

def upsamp_concat_block(x, xskip):
    u = UpSampling2D((2, 2))(x)
    return Concatenate()([u, xskip])

def res_unet(sz, batch_size):
    f = batch_size
    size = sz + (3,)
    inputs = Input(size)
    ## downsample
    e1 = bottleneck_block(inputs, f); f = int(f*2)
    e2 = res_block(e1, f, strides=2); f = int(f*2)
    e3 = res_block(e2, f, strides=2); f = int(f*2)
    e4 = res_block(e3, f, strides=2); f = int(f*2)
    _ = res_block(e4, f, strides=2)

    ## bottleneck
    b0 = conv_block(_, f, strides=1)
    _ = conv_block(b0, f, strides=1)

    ## upsample
    _ = upsamp_concat_block(_, e4)
    _ = res_block(_, f); f = int(f/2)

    _ = upsamp_concat_block(_, e3)
    _ = res_block(_, f); f = int(f/2)

    _ = upsamp_concat_block(_, e2)
    _ = res_block(_, f); f = int(f/2)

    _ = upsamp_concat_block(_, e1)
    _ = res_block(_, f)

    ## classify
    outputs = Conv2D(1, (1, 1), padding="same", activation="sigmoid")(_)

    #model creation
    model = Model(inputs=[inputs], outputs=[outputs])
    return model


'''
********************************************UTILITIES*******************************************
'''

def visualize(image, mask, original_image=None, original_mask=None):
    '''This function allows you to visualize a image and mask pair
    '''
    fontsize = 18
    mask = mask.squeeze() # comment out this line if you get an error with mask
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



def split_test_train_val(image_dir,label_dir,split = .8):
    '''
    splits images into 'Train' &  'TEST', then splits 'Train' into 'TRAIN' & 'VAL'
    inputs = image_dir (jpgs), label_dir(pngs)
    directorys must contain equal number of files, image-label pairs
    '''
    ALL_images_list = sorted(os.listdir(image_dir))
    ALL_labels_list = sorted(os.listdir(label_dir))
    X_train_list, X_TEST_list, Y_train_list, Y_TEST_list = train_test_split(ALL_images_list, ALL_labels_list,train_size = split, random_state = 2)
    X_TRAIN_list, X_VAL_list, Y_TRAIN_list, Y_VAL_list = train_test_split(X_train_list,Y_train_list,train_size = split, random_state = 3)
    TEST_images = []
    for image in X_TEST_list:
        full_path_image =image_dir + os.sep + image
        TEST_images.append(full_path_image)

    TEST_labels = []
    for label in Y_TEST_list:
        full_path_label = label_dir + os.sep + label
        TEST_labels.append(full_path_label)

    TRAIN_images = []
    for image in X_TRAIN_list:
        full_path_image = image_dir + os.sep + image
        TRAIN_images.append(full_path_image)

    TRAIN_labels = []
    for label in Y_TRAIN_list:
        full_path_label = label_dir + os.sep + label
        TRAIN_labels.append(full_path_label)

    VAL_images = []
    for image in X_VAL_list:
        full_path_image = image_dir + os.sep + image
        VAL_images.append(full_path_image)


    VAL_labels = []
    for label in Y_VAL_list:
        full_path_label = label_dir + os.sep + label
        VAL_labels.append(full_path_label)

    return TEST_images,TEST_labels,TRAIN_images,TRAIN_labels,VAL_images,VAL_labels

    ##############################################################################
def image_generator(image_files, label_files ,batch_size, sz):
  while True:
    #extract a random batch of image files
    batch_index = np.random.choice(len(image_files), size = batch_size, replace = False)

    #variables for collecting batches of inputs and outputs
    batch_x = []
    batch_y = []

    for i in batch_index:

        #get the masks. Note that masks are png files
        mask = Image.open(glob(label_files[i])[0])
        mask = np.array(mask.resize(sz))
        mask[mask == 0 ] = 0
        mask[mask > 0] = 1

        batch_y.append(mask)

        #preprocess the raw images
        raw = Image.open(image_files[i])
        raw = raw.resize(sz)
        raw = np.array(raw)

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

    yield (batch_x, batch_y)

# same image generator you know and love but now with augmentation!!!!!!
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
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.3),
        A.OneOf([
            A.OpticalDistortion(p=0.3),
            A.GridDistortion(p=.1),
            A.IAAPiecewiseAffine(p=0.3),
        ], p=0.2),
        A.OneOf([
            A.CLAHE(clip_limit=2),
            A.RandomBrightnessContrast()], p = 0.4)])

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

###########################################################################

# accuracy metrics

def mean_iou(y_true, y_pred):
    yt0 = y_true[:,:,:,0]
    yp0 = tf.keras.backend.cast(y_pred[:,:,:,0] > 0.5, 'float32')
    inter = tf.math.count_nonzero(tf.logical_and(tf.equal(yt0, 1), tf.equal(yp0, 1)))
    union = tf.math.count_nonzero(tf.add(yt0, yp0))
    iou = tf.where(tf.equal(union, 0), 1., tf.cast(inter/union, 'float32'))
    return iou

#function to define how the dice coefficient is calculated
def dice_coef(y_true, y_pred):
    smooth = 1
    y_true_f = tf.reshape(tf.dtypes.cast(y_true, tf.float32), [-1])
    y_pred_f = tf.reshape(tf.dtypes.cast(y_pred, tf.float32), [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

# function to calculate the dice_coefficient of the label output compared to the true label
def dice_coef_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)


# program to load models
def load_models(model_directory, exts = ['.h5', '.H5'], optimizer = 'adam', loss = dice_coef_loss, batch_sz = 5):
    filelist = os.listdir(model_directory)
    model_list = [f for f in filelist if f.endswith(tuple(exts))]
    print(f'{len(model_list)} Models in model directory:{model_list}')
    n = 0
    model_dict = model_dictionary()
    for m in model_list:
        n +=1
        model_name = 'model_' + str(n)
        print(f'loading {model_name}:{m}........')
        model = res_unet(sz,batch_sz)
        model.compile(optimizer = optimizer, loss = loss, metrics = [dice_coef ,'acc'])
        print("compiling model...")
        model.load_weights(model_directory + os.sep + m)
        print("lodaing weights....")
        model_dict.add(model_name, model)
        print("adding to dictionary")
    print("Dictionary of models created")
    return(model_dict)

# plot model training history
def plot__history_metric(history_obj, metric, save = False, fig_run_name = 'No_name'):
  plt.plot(history_obj.history[metric])
  plt.plot(history_obj.history['val_' + metric])
  plt.title(f"model {metric}")
  plt.ylabel(metric)
  plt.xlabel('epoch')
  plt.legend(['train','val'])
  save_name = str(fig_run_name + '_' + metric + '.png')
  cwd = os.getcwd()
  if save:
    plt.savefig(save_name)
    print(f"figure {save_name} saved to {cwd}")
  else:
    print(f'Output: {save_name} shown, but not saved')


def plot_history_diceloss_and_loss(history_obj, save = False, fig_run_name = 'No_name'):
    '''This function plots diceloss and loss and gives an option to save the outputs
    '''
    #plot 1
    plt.figure(figsize=(12,10))
    plt.subplot(121)
    plt.plot(history_obj.history['loss'])
    plt.plot(history_obj.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='lower right')
    # plot 2
    plt.subplot(122)
    plt.plot(history_obj.history['dice_coef'])
    plt.plot(history_obj.history['val_dice_coef'])
    plt.title('model dice coefficient')
    plt.ylabel('Dice coefficient')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper right')
    plt.subplots_adjust(top=0.1, bottom=0.08, left=0.10, right=0.95, hspace=0.25,wspace=0.35)
    plt.tight_layout()
    plt.show()
    save_name = str(fig_run_name + '_diceloss_and_loss.png')
    cwd = os.getcwd()
    if save:
        plt.savefig(save_name)
        print(f"figure {save_name} saved to {cwd}")
    else:
        print(f'Output: {save_name} shown, but not saved')

def plot_history_all(history_obj, save = False, fig_run_name = 'No_name'):
    '''This function plots diceloss and loss and gives an option to save the outputs
    '''
    #plot 1
    plt.figure(figsize=(12,10))
    plt.subplot(121)
    plt.plot(history_obj.history['loss'])
    plt.plot(history_obj.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper right')
    # plot 2
    plt.subplot(122)
    plt.plot(history_obj.history['dice_coef'])
    plt.plot(history_obj.history['val_dice_coef'])
    plt.title('model dice coefficient')
    plt.ylabel('Dice coefficient')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='lower right')
    plt.tight_layout()
    # plot 3
    # plt.subplot(223)
    # plt.plot(history_obj.history['binary_accuracy'])
    # plt.plot(history_obj.history['val_binary_accuracy'])
    # plt.title('model binary accuracy')
    # plt.ylabel('binary accuracy')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'val'], loc='lower right')
    # plt.tight_layout(pad=3.0)



    save_name = str(fig_run_name + '.png')
    cwd = os.getcwd()
    if save:
        plt.gcf()
        plt.savefig(save_name, format="png")
        print(f"figure {save_name} saved to {cwd}")
        plt.show(block = False)
        plt.pause(4)
        plt.close()

    else:
        print(f'Output: {save_name} shown, but not saved')
        plt.show(block = False)
        plt.pause(4)
        plt.close()




#######################################################################

def evaluate_model_accuarcy(test_generator,model, threshold = 0.5):
    ''' This function goes through all of the testing images.
    it predicts them, plots, and determines precision, recall, and f1 score
    returns three objects, mean_f1, mean_precision, mean_recall
    '''
    x, y = next(test_generator)
    y_pred = []
    y_true = []
    f1_scores_binary = []
    f1_scores_micro = []
    precision_scores = []
    recall_scores = []
    for i in range(0, len(x)):
        raw = x[i]
        raw = raw[:,:,0:3]
        pred = model.predict(np.expand_dims(raw, 0)) # create a dimension of zeros to populate in predict
        #mask post-processing
        msk  = pred.squeeze() # remove a one-dimensional layer?
        msk = np.stack((msk,)*3, axis=-1) # grab the last layer = mask layer? or the B layer?
        msk[msk >= threshold] = 1
        msk[msk < threshold] = 0
        y_true.append(y[i].squeeze())
        y_pred.append(msk[:,:,0])
        f1_score_mic = sklearn.metrics.f1_score( y_pred[i],y_true[i], labels=None, pos_label=1, average='micro', sample_weight=None, zero_division='warn')
        f1_scores_micro.append(f1_score_mic)
        precision_score = sklearn.metrics.precision_score( y_pred[i],y_true[i], labels=None, pos_label=1, average='micro', sample_weight=None, zero_division='warn')
        recall_score = sklearn.metrics.recall_score( y_pred[i],y_true[i], labels=None, pos_label=1, average='micro', sample_weight=None, zero_division='warn')
        precision_scores.append(precision_score)
        recall_scores.append(recall_score)
        y_tf = y_true[i].flatten()
        y_p = y_pred[i].astype('uint8')
        y_pf = y_p.flatten()
        f1_score_bin = sklearn.metrics.f1_score(y_pf, y_tf,average = 'binary', pos_label = 1)
        f1_scores_binary.append(f1_score_bin)
        combined = np.concatenate([raw, msk, raw* msk], axis = 1)
        title = 'f1_score_bin:' + str(f1_score_bin)
        plt.title(title)
        plt.axis('off')
        plt.imshow(combined)

        if (f1_score_bin < .5):
            name1 = "poor_prediction" + str(i) +'_f1_bin_' +str(f1_score_bin)+ '.PNG'
            plt.gcf()
            plt.savefig(name1)
            plt.show(block = False)
            plt.pause(3)
            plt.close()
        elif (f1_score_bin > .97):
            name2 = "Excellent_prediction" + str(i) + '_f1_bin_' +str(f1_score_bin)+'.PNG'
            plt.gcf()
            plt.savefig(name2)
            plt.show(block = False)
            plt.pause(3)
            plt.close()
        else:
            plt.gcf()
            plt.show(block = False)
            plt.pause(3)
            plt.close()
        print('f1_score_binary:',f1_score_bin)
        print('f1_score_micro:', f1_score_mic)
        print('precision:',precision_score)
        print('recall:',recall_score)

    mean_f1_micro = mean(f1_scores_micro)
    mean_f1_binary = mean(f1_scores_binary)
    mean_precision = mean(precision_scores)
    mean_recall = mean(recall_scores)
    n_images = len(y_pred)
    print(f'Mean f1 score micro for {n_images} testing images:{mean_f1_micro}')
    print(f'Mean f1 score binary for {n_images} testing images:{mean_f1_binary}')
    print(f'Mean precision for {n_images} testing images:{mean_precision}' )
    print(f'Mean recall for {n_images} testing images:{mean_recall}')
    return mean_f1_micro, mean_f1_binary, mean_precision, mean_recall


def get_avg_f1(model,TEST_images, TEST_labels, threshold = 0.5):

    test_generator_f1 = image_generator(TEST_images, TEST_labels, batch_size= len(TEST_images), sz = sz)
    x, y = next(test_generator_f1)
    y_pred = model.predict(x)
    y_pred[y_pred >=threshold] = 1
    y_pred[y_pred < threshold] = 0
    y_pred = y_pred.squeeze()
    y_pred = np.array(y_pred)
    y_true = y.squeeze()
    y_true = np.array(y_true)
    f1_scores = []
    for i in range(0,len(y_true)):
        f1 = f1_score(y_pred[i], y_true[i], average = 'micro')
        print(f"score: {f1}")
        f1_scores.append(f1)
    avg_f1_score = np.mean(f1_scores)
    print(f"Average F1-macro score on Test set: {avg_f1_score}")
    return f1_scores, avg_f1_score

##################################################

def img_lab_to_list_path(img_dir, lab_dir):
    T1x = []
    valid_extensions = ['.JPG', '.jpg']
    path = img_dir
    for filename in os.listdir(path):
        ext = os.path.splitext(filename)[1]
        if ext.lower() not in valid_extensions:
            continue
        T1x.append(os.path.join(path,filename))
        T1x = sorted(T1x)
    T1y = []
    valid_lab_extensions = ['.PNG','.png']
    path = lab_dir
    for filename in os.listdir(path):
        ext = os.path.splitext(filename)[1]
        if ext.lower() not in valid_lab_extensions:
            continue
        T1y.append(os.path.join(path,filename))
        T1y = sorted(T1y)
    images = np.array(T1x)
    labels = np.array(T1y)
    for image,label in zip(images,labels):
        print(f' {image} : - : {label}')
    return images, labels

def img_lab_to_list(img_dir, lab_dir):
    T1x = []
    valid_extensions = ['.JPG', '.jpg']
    path = img_dir
    for filename in os.listdir(path):
        ext = os.path.splitext(filename)[1]
        if ext.lower() not in valid_extensions:
            continue
        T1x.append(filename)
        T1x = sorted(T1x)
    T1y = []
    valid_lab_extensions = ['.PNG','.png']
    path = lab_dir
    for filename in os.listdir(path):
        ext = os.path.splitext(filename)[1]
        if ext.lower() not in valid_lab_extensions:
            continue
        T1y.append(filename)
        T1y = sorted(T1y)
    images = np.array(T1x)
    labels = np.array(T1y)
    for image,label in zip(images,labels):
        print(f' {image} : - : {label}')
    return images, labels


def test_dir_to__test_generator(test_img_dir, test_lab_dir, sz):
    T1x = []
    valid_extensions = ['.JPG', '.jpg']
    path = test_img_dir
    for filename in os.listdir(path):
        ext = os.path.splitext(filename)[1]
        if ext.lower() not in valid_extensions:
            continue
        T1x.append(os.path.join(path,filename))
        T1x = sorted(T1x)
    T1y = []
    valid_lab_extensions = ['.PNG','.png']
    path = test_lab_dir
    for filename in os.listdir(path):
        ext = os.path.splitext(filename)[1]
        if ext.lower() not in valid_lab_extensions:
            continue
        T1y.append(os.path.join(path,filename))
        T1y = sorted(T1y)
    Test_images_1 = np.array(T1x)
    Test_labels_1 = np.array(T1y)
    for image,label in zip(Test_images_1,Test_labels_1):
        print(f' {image} : - : {label}')
    test_generator = image_generator(Test_images_1,Test_labels_1,batch_size = len(Test_images_1),sz = sz)
    return test_generator

import json,codecs
import numpy as np
def saveHist(path,history):

    new_hist = {}
    for key in list(history.history.keys()):
        if type(history.history[key]) == np.ndarray:
            new_hist[key] = history.history[key].tolist()
        elif type(history.history[key]) == list:
           if  type(history.history[key][0]) == np.float64:
               new_hist[key] = list(map(float, history.history[key]))

    print(new_hist)
    with codecs.open(path, 'w', encoding='utf-8') as file:
        json.dump(new_hist, file, separators=(',', ':'), sort_keys=True, indent=4)

def loadHist(path):
    with codecs.open(path, 'r', encoding='utf-8') as file:
        n = json.loads(file.read())
    return n

def describe(array):
  print(f'Describing array')
  print(f'shape = {array.shape}')
  print(f'dtype = {array.dtype}')
  print(f'ndims = {array.ndim}')
  print(f'Unique values = {np.unique(array)}')

# def crf_postprocessing(input_image, predicted_labels, num_classes):
#
#     compat_spat=10
#     compat_col=100
#     theta_spat = 1
#     theta_col = 100
#     num_iter = 10
#
#     h, w = input_image.shape[:2]
#
#     d = densecrf.DenseCRF2D(w, h, 2)
#
#     # For the predictions, densecrf needs
#     predicted_unary = unary_from_labels(predicted_labels, num_classes, gt_prob= 0.51)
#     d.setUnaryEnergy(predicted_unary)
#
#     # densecrf takes into account additional features to refine the predicted label maps.
#     # First, as explained in the `pydensecrf` repo, we add the color-independent term,
#     # where features are the locations only:
#     d.addPairwiseGaussian(sxy=(theta_spat, theta_spat), compat=compat_spat, kernel=densecrf.DIAG_KERNEL,
#                           normalization=densecrf.NORMALIZE_SYMMETRIC)
#
#     # Then we add the color-dependent term, i.e. features are (x,y,r,g,b) based on the input image:
#     input_image_uint = (input_image*255).astype(np.uint8)
#     d.addPairwiseBilateral(sxy=(theta_spat, theta_spat), srgb=(theta_col, theta_col, theta_col), rgbim=input_image_uint,
#                            compat=compat_col, kernel=densecrf.DIAG_KERNEL,
#                            normalization=densecrf.NORMALIZE_SYMMETRIC)
#
#     # Finally, we run inference to obtain the refined predictions:
#     refined_predictions = np.array(d.inference(num_iter)).reshape(num_classes, h, w)
#
#     return np.argmax(refined_predictions,axis=0)
