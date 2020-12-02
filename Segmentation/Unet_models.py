'''Unet models for binary semantic segmentation of images 128x128 pixels


'''

# imports


#%tensorflow_version 2.x
import tensorflow as tf

from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, Dropout
from tensorflow.keras.layers import Concatenate, Conv2DTranspose, Flatten, Activation, Add, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Conv2D, Conv2DTranspose
from tensorflow.keras.constraints import max_norm
from tensorflow.keras import backend as K
import numpy as np

# image size
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
