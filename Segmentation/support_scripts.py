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
import albumentations as A

# from pydensecrf import densecrf
# from pydensecrf.utils import unary_from_labels

def split_test_train_val(image_dir,label_dir,split = .8):
    ''' inputs = image_dir (jpgs), label_dir(pngs)
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
    plt.subplot(221)
    plt.plot(history_obj.history['loss'])
    plt.plot(history_obj.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper right')
    # plot 2
    plt.subplot(222)
    plt.plot(history_obj.history['dice_coef'])
    plt.plot(history_obj.history['val_dice_coef'])
    plt.title('model dice coefficient')
    plt.ylabel('Dice coefficient')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='lower right')

    plt.tight_layout()
    # plot 3
    plt.subplot(223)
    plt.plot(history_obj.history['binary_accuracy'])
    plt.plot(history_obj.history['val_binary_accuracy'])
    plt.title('model binary accuracy')
    plt.ylabel('binary accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='lower right')
    plt.tight_layout(pad=3.0)



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
