# A program to undistort images using the information from the matlab camera calibration
#
site = 'RC0220Ra'
calibration = 'cal2018'
import os
import scipy as sp
import glob
import numpy as np
#from imageio import imread
#import imreg_dft as ird
import matplotlib.pyplot as plt
import cv2
import datetime as dt

startTime = dt.datetime.now()
# reference image, or the image all others will be registered to
#reference_img = "reference_image/RC0307Rf_20171002_1129.JPG"

ct = dt.datetime.now()
year = str(ct.year)
month = str(ct.month)
day = str(ct.day)

out_dir = ("undistorted_images"+ year +"_"+ month +"_" + day)
os.makedirs(out_dir)

rootfolder = os.getcwd() # where images are located, by default its current working directory
filelist = os.listdir(rootfolder)
files_jpg = [f for f in filelist if f.endswith((".jpg",".JPG"))]

# the 'A' matrix or camera intrinsic matrix contains the focal length, the principal point, and the skew of the camera
#irrespective of the scene
A_Mat = np.array([[3876.05658615236,0,0],[0,3895.72930299343,0],[1913.44438724644,1371.72465447568,1]])
print("A matrix from matlab:\n", A_Mat)

A = A_Mat.T


# from matlab output
# skew = 0
# tangential distortion = 0
# radial distortion =
k1 = -0.103855357514449
k2 = 0.0917758599801266
p1 = 0
p2 = 0
k3 = 0
distCoeff = np.array([k1,k2,p1,p2,k3])
print( "distortion coefficients:\n", distCoeff)

for f in files_jpg:
    (filename, ext) = os.path.splitext(f)
    print(f"undistorting {filename} using {calibration} from {site}")
    img = cv2.imread(f)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    newcameramatx, roi = cv2.getOptimalNewCameraMatrix(A,distCoeff,(w,h),1,(w,h))
    dst = cv2.undistort(img,A,distCoeff,None,newcameramatx)
    dst = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    cv2.imwrite(os.path.join(out_dir,filename + "_und" + ext),dst)
    now = dt.datetime.now() - startTime
    print(filename, 'coverted',now)
end = dt.datetime.now()
elapsed_time = end - ct
print("Processing Finished, time elsapsed:",elapsed_time)
print(len(files_jpg),"files converted")
