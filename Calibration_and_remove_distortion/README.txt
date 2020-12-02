Camera calibration was conducted on the 2018 and 2019 trips, though all of the sites here were calibrated on the 2018 trips


Geometric camera calibration, also referred to as camera resectioning, 
estimates the parameters of a lens and image sensor of an image or video camera. 
You can use these parameters to correct for lens distortion, measure the size of an object in world units, 
or determine the location of the camera in the scene. 
These tasks are used in applications such as machine vision to detect and measure objects. 
They are also used in robotics, for navigation systems, and 3-D scene reconstruction

Camera parameters include intrinsics, extrinsics, and distortion coefficients. 
To estimate the camera parameters, you need to have 3-D world points and their corresponding 2-D image points.
 You can get these correspondences using multiple images of a calibration pattern, such as a checkerboard. 
 Using the correspondences, you can solve for the camera parameters. After you calibrate a camera,
 to evaluate the accuracy of the estimated parameters, you can:
		Plot the relative locations of the camera and the calibration pattern
		Calculate the reprojection errors.
		Calculate the parameter estimation errors.

The calibration algorithm calculates the camera matrix using the extrinsic and intrinsic parameters. 
The extrinsic parameters represent a rigid transformation from 3-D world coordinate system to the 3-D camera’s
 coordinate system. The intrinsic parameters represent a projective transformation from the 3-D camera’s 
 coordinates into the 2-D image coordinates.
 
 For this camera calibration (2018) we used a 10x10 inch aluminum checker board with
 6x6 checkers and 5x5 internal corners, Each checker is 42.33mm in width
 
 The 2019 calibration made use of a 9x7 asymmetric pattern instead, because matlab's camera calibration software
 wants a assymetrical checker board.
 
 Calibrations were all done using Matlab's camera calibration module.
 I found opencv's camera calibration software difficult to use thus I utilized Matlab.
 
 Calibrations:
 
 RC0220Ra:
 Of the 18 images taken,12 images were detected by matlab, 6 were rejected. 
 Overall mean error for this calibration was .71 pixels
 
 RC0307Rf: 
 of the 27 images taken, 6 were detected by matlab and 21 were rejected.
 Overall mean error for this calibration was .30 pixels
 
 RC1227R:
 of the 35 images taken, 14 were detected by matlab and 21 were rejected.
 overall mean error for this calibration was .85 pixels 
 
 RC1459L:
 of the 32 images taken, 23 were detected by matlab and 9 were rejected.
 overall mean error for this calibration was .60 pixels 
 
 
 
 
 