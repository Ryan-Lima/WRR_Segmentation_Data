from __future__ import division
# image registration




import os
import glob
import datetime
import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
#!pip install imreg_dft
import imreg_dft as ird

import numpy.ma as ma
import imageio
from skimage import img_as_float
from scipy.interpolate import griddata


# functions for creating new directories for output
ct = datetime.datetime.now() # current time
year = str(ct.year)
month = str(ct.month)
day = str(ct.day)


# Loading reference images
root  = os.getcwd() # you might have to change this depending on where you run the scripts from
ref_img_dir = root + os.sep + 'reference_images' + os.sep

reference_img_dict = {'RC0220Ra_2017_ref_img_path': ref_img_dir + os.sep + 'RC0220Ra_20171001_1341_und.JPG',
    'RC0220Ra_2019_ref_img_path' : ref_img_dir + os.sep + 'RC0220Ra_20191006_1414_und.JPG' ,
    'RC0235L_2018_ref_img_path' : ref_img_dir + os.sep +'RC0235L_20180927_1502_und.JPG',
    'RC0307Rf_2017_ref_img_path' : ref_img_dir + os.sep +'RC0307Rf_20171002_1129_und.JPG',
    'RC0307Rf_2019_ref_img_path' : ref_img_dir + os.sep +'RC0307Rf_20191007_1145_und.JPG',
    'RC0566R_2018_ref_img_path' : ref_img_dir + os.sep +'RC0566R_20181001_1152_und.JPG',
    'RC0917R_2018_ref_img_path' : ref_img_dir + os.sep +'RC0917R_20181004_1346_und.JPG',
    'RC1044R_2018_ref_img_path' : ref_img_dir + os.sep +'RC1044R_20181005_1203_und.JPG',
    'RC1227R_2017_ref_img_path' : ref_img_dir + os.sep +'RC1227R_20171010_1513_und.JPG',
    'RC1227R_2019_ref_img_path' : ref_img_dir + os.sep +'RC1227R_20191015_1712_und.JPG',
    'RC1377L_2018_ref_img_path' : ref_img_dir + os.sep +'RC1377L_20181007_1457_und.JPG',
    'RC1459L_2017_ref_img_path' : ref_img_dir + os.sep +'RC1459L_20171012_1509_und.JPG',
    'RC1459L_2019_ref_img_path' : ref_img_dir + os.sep +'RC1459L_20191017_1242_und.JPG',
    'RC1946L_2017_ref_img_path' : ref_img_dir + os.sep +'RC1946L_20171014_1511_und.JPG'}

def get_image_list_from_path(path, exts = ['.JPG','.jpg']):
    filelist = os.listdir(path)
    images_list = [f for f in filelist if f.endswith(tuple(exts))]
    print(f'files in {path}:\n {images_list}')
    return images_list

def cv2_to_PILimg(cv2img):
    img = cv2.cvtColor(cv2img, cv2.COLOR_BGR2RGB)
    PIL_img = Image.fromarray(img)
    return PIL_img

def PIL_to_cv2img(PILimg):
    cv2img = np.asarray(PILimg)
    return cv2img

def show_images(images, cols = 1, titles = None):
    """Display a list of images in a single figure with matplotlib.

    Parameters
    ---------
    images: List of np.arrays compatible with plt.imshow.

    cols (Default = 1): Number of columns in figure (number of rows is
                        set to np.ceil(n_images/float(cols))).

    titles: List of titles corresponding to each image. Must have
            the same length as titles.
    """
    assert((titles is None)or (len(images) == len(titles)))
    n_images = len(images)
    if titles is None: titles = ['Image (%d)' % i for i in range(1,n_images + 1)]
    fig = plt.figure()
    for n, (image, title) in enumerate(zip(images, titles)):
        a = fig.add_subplot(cols, np.ceil(n_images/float(cols)), n + 1)
        if image.ndim == 2:
            plt.gray()
        plt.imshow(image)
        a.set_title(title)
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
    plt.show(block = False)
    plt.pause(10)
    plt.close()

# def create_reg_dir(working_directory):
#     wd = working_directory
#     ct = datetime.datetime.now() # current time
#     year = str(ct.year)
#     month = str(ct.month)
#     day = str(ct.day)
#     reg_dir = wd + os.sep + 'registered_' + year + "_" + month + "_" + day
#     if not os.path.exists(reg_dir):
#         os.makedirs(reg_dir)
#     return reg_dir

def img_path_to_cv2BGR_RGB_GRAY(image_path):
    print('image_path',image_path)
    img_bgr = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    img_PIL = Image.open(image_path)
    return img_bgr, img_rgb, img_gray, img_PIL

def register_dft(ref_image_path, sub_image_filename, sub_image_in_dir, reg_dir,unable_to_reg_list,numiter = 15):
    '''
    This is a fast and relatively accurate form of registration which makes uses of imreg_dft library
    This method is preferred but only results in translation not homography. This method also returns a border
    which is not black, which might be problematic.
    '''
    print('attempting dft registration:')
    filename, ext = os.path.splitext(sub_image_filename)
    print(f'reference image = {ref_image_path}')
    print(f'beginning registration on {sub_image_filename}')
    rim_bgr, rim_rgb, rim_gray, rim_PIL = img_path_to_cv2BGR_RGB_GRAY(ref_image_path)
    im0 = rim_gray
    sim_bgr, sim_rgb, sim_gray, sim_PIL = img_path_to_cv2BGR_RGB_GRAY(os.path.join(sub_image_in_dir,sub_image_filename))
    im1 = sim_gray
    start = datetime.datetime.now()
    #try:
    result = ird.similarity(im0, im1, numiter=numiter)
    new_name = filename + '_regdft' + ext
    assert "timg" in result
    try:# Maybe we don't want to show plots all the time
        if os.environ.get("IMSHOW", "yes") == "yes":
            tvec = result['tvec']
            ird.imshow(im0, im1, result['timg'])
            plt.show(block = False )
            plt.savefig(filename + '_regdft_plot.png')
            plt.pause(5)
            plt.close()
            print("Translation is {}, success rate {:.4g}".format(tuple(tvec), result["success"]))
        t = result['tvec']
        a = result['angle']
        s = result['scale']
        timg = ird.transform_img(sim_bgr, angle = a, tvec = t, scale = s)
        if os.environ.get("IMSHOW", "yes") == "yes":
            plt.figure()
            plt.imshow(timg)
            plt.show(block = False )
            plt.pause(5)
            plt.close()
        cv2.imwrite(os.path.join(reg_dir,new_name),timg)
        end = datetime.datetime.now()
        print(f'image {filename} registered and saved as {new_name} in {reg_dir}' )
        elapsed = end - start
        print(f'registration took {elapsed}')
        registered_image_full_path = os.path.join(reg_dir,new_name)
        return(registered_image_full_path)
    except Exception:
        unable_to_reg_list.append(sub_image_filename)
        end = datetime.datetime.now()
        elapsed = end - start
        print(f'unable to register {sub_image_filename} and it took {elapsed}')
        return(registered_image_full_path)

def register_ECC(ref_image_path, sub_image_filename, sub_image_in_dir, reg_dir,unable_to_reg_list, numiter = 100, warp_mode = cv2.MOTION_HOMOGRAPHY):
    '''This registration scripts makes used of the Enhanced correlation coefficient. This workflow is described in a post
    https://www.learnopencv.com/image-alignment-ecc-in-opencv-c-python/
    and described in this paper:
    http://xanthippi.ceid.upatras.gr/people/evangelidis/george_files/PAMI_2008.pdf
    'Evangelidis and Psarakis 2008'

    pretty slow and sort of accurate

    This mode by default uses homography and might be the registration we choose when the camera has shifted
    since it is not just translating the image but calculating a homography. Its accuracy varies, use test_registration_stack() to
    visualize the quality of the registration.

    '''
    print('attempting ECC registration:')
    filename, ext = os.path.splitext(sub_image_filename)
    new_name = filename + '_regECC'+ ext
    new_filepath = os.path.join(reg_dir, new_name)
    rim_bgr, rim_rgb, rim_gray, rim_PIL= img_path_to_cv2BGR_RGB_GRAY(ref_image_path)
    im0 = rim_gray
    sub_image_path = os.path.join(sub_image_in_dir,sub_image_filename)
    sim_bgr, sim_rgb, sim_gray, sim_PIL = img_path_to_cv2BGR_RGB_GRAY(sub_image_path)
    im1 = sim_gray
    h, w = sim_rgb.shape[:2]
    # Define 2x3 or 3x3 matrices and initialize the matrix to identity
    if warp_mode == cv2.MOTION_HOMOGRAPHY :
        warp_matrix = np.eye(3, 3, dtype=np.float32)
    else :
        warp_matrix = np.eye(2, 3, dtype=np.float32)

    # Specify the number of iterations.
    print(f'begin iterations on {filename}')
    number_of_iterations = numiter;

    # Specify the threshold of the increment
    # in the correlation coefficient between two iterations
    termination_eps = 1e-10;

    registered_image_full_path = new_filepath
    # Define termination criteria

    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations,  termination_eps)
    try:
        (cc, warp_matrix) = cv2.findTransformECC (im0,im1,warp_matrix, warp_mode, criteria, None, 1)

        if warp_mode == cv2.MOTION_HOMOGRAPHY :
        # Use warpPerspective for Homography
            im1_aligned = cv2.warpPerspective (sim_bgr, warp_matrix, (w,h), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
        else :
        # Use warpAffine for Translation, Euclidean and Affine
            im1_aligned = cv2.warpAffine(sim_bgr, warp_matrix, (w,h), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP);
        if not os.path.isfile(new_filepath):
            cv2.imwrite(new_filepath,im1_aligned)
        print(f'registered image {filename}')

        return(registered_image_full_path)
        # I don't think this is needed MKF 11/10/2020
        # aligned_rgb = cv2.cvtColor(im1_aligned, cv2.COLOR_BGR2RGB)
        # images = [rim_rgb, sim_rgb, aligned_rgb]
        # titles = ['Reference Image', 'Subject Image', 'Registered Image']
        # show_images(images, titles = titles)

    except Exception:
        print(f'Unable to register {registered_image_full_path }')
        unable_to_reg_list.append(sub_image_filename)
        return(registered_image_full_path)  # or you could use 'continue'

def register_ECC_Crop(ref_image_path, sub_image_filename, sub_image_in_dir, reg_dir,unable_to_reg_list,numiter = 1000, warp_mode = cv2.MOTION_TRANSLATION, crop = 250):
    '''
    This registration scripts makes used of the Enhanced correlation coefficient. This workflow is described in a post
    https://www.learnopencv.com/image-alignment-ecc-in-opencv-c-python/
    and described in this paper:
    http://xanthippi.ceid.upatras.gr/people/evangelidis/george_files/PAMI_2008.pdf
    'Evangelidis and Psarakis 2008'

    pretty slow and sort of accurate
    use 'test_registration_stack()' to visualize registration quality.
    This utilizes a motion translation or shift in X and Y by default but can be changed to a homography
    This version also makes uses of a crop prior to registration.
    it uses the user defined crop = default 250 to crop in 250 pixels from each edge before registration.
    This can be useful if the undistortion process caused large black edges around  the image which
    may throw off the registration

    '''
    print('attempting ECC registration with crop:')
    filename, ext = os.path.splitext(sub_image_filename)
    new_name = filename + '_regECCcrop'+ ext
    new_filepath = os.path.join(reg_dir, new_name)
    rim_bgr, rim_rgb, rim_gray, rim_PIL= img_path_to_cv2BGR_RGB_GRAY(ref_image_path)
    sub_image_path = os.path.join(sub_image_in_dir,sub_image_filename)
    sim_bgr, sim_rgb, sim_gray, sim_PIL = img_path_to_cv2BGR_RGB_GRAY(sub_image_path)
    simh, simw = sim_rgb.shape[:2]
    rimh, rimw = rim_bgr.shape[:2]

    left_x1 = crop
    top_y1 = crop
    right_x2 = rimw - crop
    bottom_y2 = rimh - crop
    rimcropPIL = rim_PIL.crop((left_x1,top_y1,right_x2,bottom_y2))
    rimcropCv2 = PIL_to_cv2img(rimcropPIL)
    im0 = cv2.cvtColor(rimcropCv2, cv2.COLOR_BGR2GRAY)

    left = crop
    top = crop
    right = simw - crop
    bottom = simh - crop
    simcropPIL = sim_PIL.crop((left, top, right, bottom))
    simcropCv2 = PIL_to_cv2img(simcropPIL)
    im1 = cv2.cvtColor(simcropCv2, cv2.COLOR_BGR2GRAY)

    # Define 2x3 or 3x3 matrices and initialize the matrix to identity
    if warp_mode == cv2.MOTION_HOMOGRAPHY :
        warp_matrix = np.eye(3, 3, dtype=np.float32)
    else :
        warp_matrix = np.eye(2, 3, dtype=np.float32)

    # Specify the number of iterations.
    print(f'begin iterations on {filename}')
    number_of_iterations = numiter;

    # Specify the threshold of the increment
    # in the correlation coefficient between two iterations
    termination_eps = 1e-10;

    # Define termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations,  termination_eps)
    registered_image_full_path = new_filepath
    try:
        (cc, warp_matrix) = cv2.findTransformECC (im0,im1,warp_matrix, warp_mode, criteria, None, 1)

        if warp_mode == cv2.MOTION_HOMOGRAPHY :
        # Use warpPerspective for Homography
            im1_aligned = cv2.warpPerspective (sim_bgr, warp_matrix, (simw,simh,), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
        else :
        # Use warpAffine for Translation, Euclidean and Affine
            im1_aligned = cv2.warpAffine(sim_bgr, warp_matrix, (simw,simh,), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP);
        if not os.path.isfile(new_filepath):
            cv2.imwrite(new_filepath,im1_aligned)
        # print(f'registered image {filename}')
        # aligned_rgb = cv2.cvtColor(im1_aligned, cv2.COLOR_BGR2RGB)
        # images = [rim_rgb, sim_rgb, aligned_rgb]
        # titles = ['Reference Image', 'Subject Image', 'Registered Image']
        # show_images(images, titles = titles)
        return(registered_image_full_path)

    except Exception:
        print(f'Unable to register {registered_image_full_path }')
        unable_to_reg_list.append(sub_image_filename)
        return(registered_image_full_path)

def register_2dfft(ref_image_path, sub_image_filename, sub_image_in_dir, reg_dir,unable_to_reg_list):
    '''
    this is a scipt to use 2D fast forrier transform to register images. This is the method described by Buscombe in Grams et al., 2018 open file report
    It is by far the fastest but not always the most accurate. use 'test_registration_stack()' to visualize the quality of the registration.
    this registration does shifts in X and Y (translation)
    '''
    print('attempting 2dfft registration:')
    filename, ext = os.path.splitext(sub_image_filename)
    print(f'reference image = {ref_image_path}')
    print(f'beginning registration on {sub_image_filename}')
    rim_bgr, rim_rgb, rim_gray, rim_PIL = img_path_to_cv2BGR_RGB_GRAY(ref_image_path)
    im0 = rim_gray
    sim_bgr, sim_rgb, sim_gray, sim_PIL = img_path_to_cv2BGR_RGB_GRAY(os.path.join(sub_image_in_dir,sub_image_filename))
    im1 = sim_gray
    start = datetime.datetime.now()
    shape = im0.shape
    new_name = filename + '_reg2dfft' + ext
    new_filepath = os.path.join(reg_dir, new_name)
    f0 = np.fft.fft2(im0)
    # compute 2D FFT of sample image
    f1 = np.fft.fft2(im1)
    # compute 2D cross-correlation function
    registered_image_full_path = new_filepath
    try:
        ir = abs(np.fft.ifft2((f0*f1.conjugate())/(abs(f0)*abs(f1))))
        # find the [x,y] location of the peak
        tx, ty = np.unravel_index(np.argmax(ir), shape)
        if tx > shape[0] //2:
           tx  -= shape[0]
        if ty > shape[1]//2:
           ty -= shape[1]

        rows, cols = im1.shape
        im1f = img_as_float(im1)

        #M = np.float32([[1,0,tx],[0,1,ty]])
        M = np.float64([[1,0,tx],[0,1,ty]])

        dst = cv2.warpAffine(sim_bgr, M,(cols,rows))
        dst_rgb = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)
        cv2.imwrite(new_filepath,dst)
        images = [rim_rgb, sim_rgb, dst_rgb]
        titles = ['reference image', 'subject image', 'registered image']
        show_images(images, titles = titles)

        return(registered_image_full_path)
    except Exception:
        print(f'Unable to register {registered_image_full_path }')
        unable_to_reg_list.append(sub_image_filename)
        return(registered_image_full_path) # if registration unsuccessfull it will return a filepath to a file that does not exist

def test_registration_stack(ref_image_path, sub_image_path ,save = False):
    ''' this function displays an image which is the average pixel value of the subject image
    or the image that has been registered and the reference image. If the image appears very blurry
    then the registration is not good, if the lines are relatively crisp then the registration is good
    This simply provides a qualitative visual estimate of registration quality.
    This function also saves a figure if save = True
    '''
    print("performing visual estimate of registraton quality")
    print(f"for image {sub_image_path}")
    filename, ext = os.path.splitext(sub_image_path)
    rim_path = ref_image_path
    rim_bgr, rim_rgb, rim_gray, rim_PIL = img_path_to_cv2BGR_RGB_GRAY(rim_path)
    sim_path = sub_image_path
    sim_bgr, sim_rgb, sim_gray, sim_PIL = img_path_to_cv2BGR_RGB_GRAY(sim_path)
    w, h = rim_PIL.size
    arr = np.zeros((h,w,3), np.float)
    im_list = [rim_path, sim_path]
    N = len(im_list)
    for im in im_list:
        imarr = np.array(Image.open(im), dtype = np.float)
        arr = arr + imarr/N
    arr = np.array(np.round(arr), dtype = np.uint8)
    out = Image.fromarray(arr, mode = 'RGB')
    plt.figure(figsize = (12,18))
    plt.title(filename)
    if save == True :
        filename = filename + "_registration_test.png"
        plt.imshow(out)
        plt.show(block = False)
        plt.savefig(filename)
        plt.pause(20)
        plt.close()
    else :
        plt.imshow(out)
        plt.show(block = False)
        plt.pause(30)
        plt.close()

def print_unable_to_register(unable_to_reg_list):
    print(f'Unable to register the following images:{unable_to_reg_list}')
    if len(unable_to_reg_list) > 0:
        with open('unable_to_register.txt', 'w') as f:
            for item in unable_to_register:
                f.write("%s\n" % item)
    else:
        print("all images registered")
