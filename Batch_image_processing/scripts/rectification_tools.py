'Script containing Sandbar rectifications and rectification testing'

# imports
from __future__ import division
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.collections
import matplotlib.patches
import numpy.ma as ma
from scipy.stats import mode
from PIL import Image, ImageDraw
import pandas as pd
import os
import random
import sys
from scipy.interpolate import griddata
import imageio


site_names_list = ["RC0220Ra","RC0307Rf", "RC1227R","RC1459L"]
years_list = ["2017","2019"]

root_dir = os.getcwd() # for google drive

Error_surface_dir = root_dir + os.sep + 'error_surfaces'

if not os.path.exists(Error_surface_dir):
    os.makedirs(Error_surface_dir)
# create the PanelDate class ClassName(object):
class PanelDate:
  ''' a panel date object contains all the information needed to create a homography for a particular
  reference image. The reference image name is located within the __str__ method of the class

  Attributes:
  impts = U,V coordinates of GCPs in the impts_panel_img
  mappts = E,N coordinates of GCPs from impts_panel_img and survey, NAD83 AZ central state plane 0202 meters
  extent_buffer = user-defined buffer in meters which is added
    and subtracted from NE_max and NE_min which provides the extent of the rectified image
  E_min = the minimum Easting value within mappts, converted to int and padded with extent_buffer
  E_max = the maximum Easting value within mappts, converted to int and padded with extent_buffer
  N_min = the minimum Northing value within mappts, converted to int and padded with extent_buffer
  N_max = the maximum Northing value within mappts, converted to int and padded with extent_buffer
  EN_min = tuple containing (E_min, N_min)
  EN_max = tuple containing (E_max, N_max)
  minpts = list containing [E_min,N_min]
  N_ext = distance in meters along the north axis of the rectified output
  E_ext = distance in meters along the east axis of the rectified output
  dsize = size of the rectified image in meters, for 1x1meter pixels
  usepts = evenly spaced spaced values the length of the impts list
  pts = mappts - minpts, cv2.findHomography has trouble with really large numbers like the 6 digit ENs
  ptsHiRes = (mappts - minpts)*10 this creates a larger image where each pixel is actually .1x.1 meters
  N_ext_HiRes = N_ext * 10
  E_ext_HiRes = E_ext * 10
  dsizeHiRes =  (E_ext_HiRes, N_ext_HiRes)
    therefore we subtract the minpts from the mappts
  impts_panel_img = the name of the panel image where the impts of GCPs were attained

  When creating a homography with 1x1 meter pixels use the following:
  H, _ = cv2.findHomography(PanelDate.impts[PanelDate.usepts], PanelDate.pts[PanelDate.usepoints])

  When creating a homography with .1x.1 meter pixels use the following:
  H, _ = cv2.findHomography(PanelDate.impts[PanelDate.usepts], PanelDate.ptsHiRes[PanelDate.usepoints])

  when applying the homography to an image use dsize for 1x1 meter homography
   and dsizeHiRes for .1x.1m homography
  '''
  def __init__(self,sitename, year, impts, mappts,panel_img_str,extent_buffer = 30):
    self.sitename = sitename
    #site_names_list = ["RC0220Ra","RC0307Rf", "RC1227R","RC1459L"]
    if sitename not in site_names_list:
      print('Error! provided sitename not in list of sites')
    self.year = year
    #years_list = ["2017","2019"]
    if year not in years_list:
      print('Error! provided year not in years list')
    self.impts = impts
    self.mappts = mappts
    self.extent_buffer = extent_buffer
    self.E_min = int(np.min(self.mappts[:,0]) - self.extent_buffer)
    self.E_minHiRes = self.E_min*10
    self.E_max = int(np.max(self.mappts[:,0]) + self.extent_buffer)
    self.E_maxHiRes = self.E_max*10
    self.N_min = int(np.min(self.mappts[:,1]) - self.extent_buffer)
    self.N_minHiRes = self.N_min*10
    self.N_max = int(np.max(self.mappts[:,1]) + self.extent_buffer)
    self.N_maxHiRes = self.N_max*10
    self.EN_min = (self.E_min, self.N_min)
    self.EN_max = (self.E_max, self.N_max)
    self.minpts = [self.E_min, self.N_min]
    self.minptsHiRes = [self.E_minHiRes, self.N_minHiRes]
    self.N_ext = (self.N_max - self.N_min)
    self.E_ext = (self.E_max - self.E_min)
    self.dsize = (self.E_ext,self.N_ext)
    self.usepts = np.arange(len(self.impts)) # evenly spaced values the length of impts
    self.pts = self.mappts - self.minpts
    self.ptsHiRes = self.pts*10
    self.N_ext_HiRes = self.N_ext*10
    self.E_ext_HiRes = self.E_ext*10
    self.dsizeHiRes = (self.E_ext_HiRes, self.N_ext_HiRes)
    self.impts_panel_img = panel_img_str


# METHODS ........................
  def compare_impts_and_mappts(self):
    '''
    This method can be run to check if the impts and mappts arrays provided are the same shape.
    impts = np.array([[u1,v1],[u2,v2],...])
    mappts = np.array([[E1,N1],[E2,N2],...])
    '''
    if self.impts.shape == self.mappts.shape:
      print(f"Arrays are same shape: {self.impts.shape}")
    else:
      print(f"Arrays have different shapes!")
    print('impts:',self.impts)
    print('mappts:', self.mappts)

  def load_homography_hires(self,H_high):
    self.homography_hires = H_high

  def load_homography_lowres(self,H_low):
    self.homography_lowres = H_low


  def __str__(self):
    return self.sitename + ':' + self.year + '\n' + str(len(self.mappts)) + " GCPs\nimpts from:" + self.impts_panel_img

## functions
def transform_points(impts, H, N_min, E_min):
  '''
  a function which transforms points using a homography

  inputs:
   H = numpy array:3x3 homography
   impts = np.array of U,V coordinates for GCPs
   N_min = the minimum northing in the rectified image - extent_buffer
   E_min = the minimum easitng in the rectified image - extent_buffer

   returns:
   transE = np.array of impts[:,0] transformed to eastings
   transN = np.array of impts[:,1] transformed to northings
  '''
  fct = 1
  x = np.array(impts[:,0])
  y = np.array(impts[:,1])
  points = np.vstack((fct*x,fct*y, np.ones(len(x))))
  txyz = H.dot(points)
  tx = txyz[0,:]
  ty = txyz[1,:]
  tz = txyz[2,:]
  tx = tx/tz
  ty = ty/tz
  txc = tx + E_min
  tyc = ty + N_min
  transE = np.array(txc)
  transN = np.array(tyc)
  return transE, transN

def rectify_hi_res(PanelDate,img_rgb, plot = True):
  H, trans = cv2.findHomography(PanelDate.impts[PanelDate.usepts],PanelDate.ptsHiRes[PanelDate.usepts])
  print(f'Homography with .1x.1 meter pixels\n {H}')
  dst = cv2.warpPerspective(img_rgb, H, PanelDate.dsizeHiRes)
  rows, cols, chanels = dst.shape
  E_axis = np.linspace(PanelDate.minptsHiRes[0], PanelDate.minptsHiRes[0] + cols, rows)
  N_axis = np.linspace(PanelDate.minptsHiRes[1], PanelDate.minptsHiRes[1] + rows, rows)
  if plot == True:
    fig = plt.figure()
    fig.subplots_adjust(wspace = 0.5)
    #plot original image
    ax = plt.subplot(1,2,1)
    plt.title("Original Image")
    plt.imshow(img_rgb)
    plt.scatter(x = PanelDate.impts[:,0],y = PanelDate.impts[:,1], c = 'r', s = 15 )
    ax.get_xaxis().get_major_formatter().set_useOffset(False)
    ax.get_yaxis().get_major_formatter().set_useOffset(False)
    labels = ax.get_xticklabels()
    plt.setp(labels, rotation=30, fontsize=8)
    labels = ax.get_yticklabels()
    plt.setp(labels, rotation=30, fontsize=8)
    # plot rectified Image
    ax = plt.subplot(1,2,2)
    plt.title("Rectified Image")
    ext =  [E_axis.min(), E_axis.max(), N_axis.min(), N_axis.max()]
    plt.imshow(dst, extent = ext, origin= 'lower')
    mapx = PanelDate.ptsHiRes[:,0] + PanelDate.minptsHiRes[0]
    mapy = PanelDate.ptsHiRes[:,1] + PanelDate.minptsHiRes[1]
    plt.scatter(x = mapx , y = mapy, c = 'r', s = 15 )
    transE, transN = transform_points(PanelDate.impts, H, PanelDate.N_minHiRes, PanelDate.E_minHiRes)
    plt.scatter(x  = transE, y = transN, c = 'b', s = 15 )

  else:
    print("Plot = False")

  return dst, H

def rectify_low_res(PanelDate,img_rgb,plot = True):
  H, trans = cv2.findHomography(PanelDate.impts[PanelDate.usepts],PanelDate.pts[PanelDate.usepts])
  print(f'Homography with 1x1 meter pixels\n {H}')
  dst = cv2.warpPerspective(img_rgb, H, PanelDate.dsize)
  rows, cols, chanels = dst.shape
  E_axis = np.linspace(PanelDate.minpts[0], PanelDate.minpts[0] + cols, rows)
  N_axis = np.linspace(PanelDate.minpts[1], PanelDate.minpts[1] + rows, rows)
  if plot == True:
    fig = plt.figure()
    fig.subplots_adjust(wspace = 0.5)
    #plot original image
    ax = plt.subplot(1,2,1)
    plt.title("Original Image")
    plt.imshow(img_rgb)
    plt.scatter(x = PanelDate.impts[:,0],y = PanelDate.impts[:,1], c = 'r', s = 15 )
    ax.get_xaxis().get_major_formatter().set_useOffset(False)
    ax.get_yaxis().get_major_formatter().set_useOffset(False)
    labels = ax.get_xticklabels()
    plt.setp(labels, rotation=30, fontsize=8)
    labels = ax.get_yticklabels()
    plt.setp(labels, rotation=30, fontsize=8)
    # plot rectified Image
    ax = plt.subplot(1,2,2)
    plt.title("Rectified Image")
    ext =  [E_axis.min(), E_axis.max(), N_axis.min(), N_axis.max()]
    plt.imshow(dst, extent = ext, origin= 'lower')
    mapx = PanelDate.pts[:,0] + PanelDate.minpts[0]
    mapy = PanelDate.pts[:,1] + PanelDate.minpts[1]
    plt.scatter(x = mapx , y = mapy, c = 'r', s = 15 )
    transE, transN = transform_points(PanelDate.impts, H, PanelDate.N_min, PanelDate.E_min)
    plt.scatter(x  = transE, y = transN, c = 'b', s = 15 )
  else:
    print("Plot = False")

  return dst, H


def batch_rectify_hi_res(PanelDate, img_list, img_dir, out_dir):
  print(f'Batch hi-res rectification for {PanelDate.sitename}')
  print(f'Using hi-res homography {PanelDate.homography_hires}')
  H = PanelDate.homography_hires
  for image in img_list:
    (filename, ext) = os.path.splitext(image)
    print(f'Rectifying {image}....')
    img_bgr= cv2.imread(img_dir + os.sep + image)
    img_rgb = cv2.cvtColor(img_bgr,cv2.COLOR_BGR2RGB)
    dst = cv2.warpPerspective(img_rgb, H, PanelDate.dsizeHiRes)
    dst_flipped = np.flip(dst,axis = 0)
    rows, cols, chanels = dst.shape
    new_name = filename + "_rectHi" + ext
    save_name = out_dir + os.sep + new_name
    if not os.path.isfile(save_name):
      imageio.imwrite(save_name, dst_flipped)
      print(f'Saving {new_name} in {out_dir}...')
    else:
      print(f'{save_name} already exists!\n rectified image not saved')


def batch_rectify_low_res(PanelDate, img_list, img_dir, out_dir):
  print(f'Batch low-res rectification for {PanelDate.sitename}')
  print(f'Using low-res homography {PanelDate.homography_lowres}')
  H = PanelDate.homography_lowres
  for image in img_list:
    (filename, ext) = os.path.splitext(image)
    print(f'Rectifying {image}....')
    img_bgr= cv2.imread(img_dir + os.sep + image)
    img_rgb = cv2.cvtColor(img_bgr,cv2.COLOR_BGR2RGB)
    dst = cv2.warpPerspective(img_rgb, H, PanelDate.dsize)
    dst_flipped = np.flip(dst,axis = 0)
    rows, cols, chanels = dst.shape
    new_name = filename + "_rectLo" + ext
    save_name = out_dir + os.sep + new_name
    if not os.path.isfile(save_name):
      imageio.imwrite(save_name, dst_flipped)
      print(f'Saving {new_name} in {out_dir}...')
    else:
      print(f'{save_name} already exists!\n rectified image not saved')

def estimate_homography_error(PanelDate, H,high_res = False):
    fct = 1
    x = PanelDate.impts[:,0]
    y = PanelDate.impts[:,1]
    points = np.vstack((fct*x, fct*y, np.ones(len(x))))
    #print(points)
    txyz = H.dot(points)
    #print(txyz)
    tx = txyz[0,:]
    ty = txyz[1,:]
    tz = txyz[2,:]
    tx = tx/tz
    ty = ty/tz
    if high_res == False:
      txc = tx + PanelDate.E_min
      tyc = ty + PanelDate.N_min
      nxE = np.array(txc)
      nyN = np.array(tyc)
      mapx = PanelDate.pts[:,0] + PanelDate.E_min
      mapy = PanelDate.pts[:,1] + PanelDate.N_min
      nNE = np.column_stack((nxE,nyN))
      #print(nNE)
      mappts_trans = np.column_stack((mapx,mapy))
      #print(mappts_trans)
      errors = nNE - mappts_trans
      ssq = np.sum(np.sqrt(errors**2),axis = 0)/len(errors)
    elif high_res == True:
      txc = tx + PanelDate.E_min*10
      tyc = ty + PanelDate.N_min*10
      nxE = np.array(txc)
      nyN = np.array(tyc)
      mapx = PanelDate.ptsHiRes[:,0] + PanelDate.E_min*10
      mapy = PanelDate.ptsHiRes[:,1] + PanelDate.N_min*10
      nNE = np.column_stack((nxE,nyN))
      #print(nNE)
      mappts_trans = np.column_stack((mapx,mapy))
      #print(mappts_trans)
      errors = nNE - mappts_trans
      errors = errors*.1
      ssq = np.sum(np.sqrt(errors**2),axis = 0)/len(errors)
    else:
      print("Error resolution not defined, choose high_res = 'True' or 'False'")
    #EmaxError = np.max(abs(error[:,0]))
    #NmaxError = np.max(abs(error[:,1]))
    print('Error all points in meters\n', errors)
    print('Sum of squared errors in all points:')
    print(f'SSE easting error {ssq[0]} meters\nSSE northing error {ssq[1]} meters')
    return errors, ssq

def minimize_homography_error(PanelDate,img_rgb,num_pts_rmvd = 1):
    if num_pts_rmvd > 4:
      print("can only remove up to 4 points, try again...")
    else:
      fct = 1
      N_points_removed = num_pts_rmvd
      N = 1000
      runs = []
      img_points = PanelDate.impts
      map_points = PanelDate.mappts
      min_points = PanelDate.minpts
      E_min = PanelDate.mappts[:,0].min()
      N_min = PanelDate.mappts[:,1].min()
      points_2 = map_points - min_points
      all_points = np.column_stack((img_points, points_2))
      # shuffle points
      for i in range(N):
          np.random.shuffle(all_points)
          removed_points = all_points[:N_points_removed]
          #print("Removed Points: ", removed_points)
          gcps_minus_removed = all_points[N_points_removed:]
          #print("Points Left: ",gcps_minus_removed)
          use_points = np.arange(len(gcps_minus_removed))
          # caluclate homography
          Homo, Stat = cv2.findHomography(gcps_minus_removed[use_points,:2],gcps_minus_removed[use_points,2:])
          # transform image points using homography to TMapx and TMapy
          x = gcps_minus_removed[:,0]
          y = gcps_minus_removed[:,1]
          points = np.vstack((fct*x, fct*y, np.ones(len(x))))
          txyz = Homo.dot(points)
          tx = txyz[0,:]
          ty = txyz[1,:]
          tz = txyz[2,:]
          tx = tx/tz
          ty = ty/tz
          txc = tx + E_min
          tyc = ty + N_min
          nxE = np.array(txc) # impts transformed using Homography E
          nxN = np.array(tyc) # impts transformed using Homography N
          Trans_image_points = np.column_stack((nxE,nxN)) # stack them together into single array
          MapE = gcps_minus_removed[:,2] + E_min # mappts - removed point
          MapN = gcps_minus_removed[:,3] + N_min # mappts - removed point
          Map_points = np.column_stack((MapE,MapN))
          error = Trans_image_points - Map_points
          sse = np.sum(np.sqrt(error**2), axis=0)/len(error)
          SSSE = np.sum(sse)
          r_p = removed_points
          run_error_dict = {'Run':i,'Sum SSE':SSSE, "Removed":removed_points.copy(), "Remaining Points":gcps_minus_removed.copy(), "Homography":Homo.copy()}
          runs.append(run_error_dict)

      Lowest_Error = {'Run': 000, 'Sum SSE':1000, 'Removed Points':[[0,0]], 'Remaining Points': [[0,0]],"Homography": [[0,0]]}
      for item in runs:
        if item['Sum SSE'] < Lowest_Error['Sum SSE']:
          Lowest_Error['Sum SSE'] = item['Sum SSE']
          Lowest_Error['Run'] = item['Run']
          Lowest_Error['Removed Points'] = item['Removed']
          Lowest_Error['Remaining Points'] = item['Remaining Points']
          Lowest_Error["Homography"] = item["Homography"]

        else:
          continue
      print(f"Lowest error {Lowest_Error['Sum SSE']} found by removing points {Lowest_Error['Removed Points']}")
    return [Lowest_Error['Removed Points'],Lowest_Error['Remaining Points']]

def plot_error_surface(PanelDate,img_rgb, out_dir, high_res = True, save = False, plot = False):
  '''
  inputs:
  > img_rgb
  > PanelDate
  > high_res = True
  > save_name
  > out_dir
  '''
  # check highres or low
  if high_res == True:
    # calculate homography
      dst, H_high = rectify_hi_res(PanelDate,img_rgb, plot = False)
      H = H_high
      E_max = PanelDate.E_maxHiRes
      N_max = PanelDate.N_maxHiRes
      minpts = PanelDate.minptsHiRes
      #print("minpts", minpts)
      E_min = minpts[0]
      N_min = minpts[1]
      pts = PanelDate.ptsHiRes
      rows, cols, channels = dst.shape
      #print("dst.shape",dst.shape)
      E_axis = np.linspace(E_min, E_min + cols, cols)
      N_axis = np.linspace(N_min, N_min + rows, rows)
      transE, transN = transform_points(PanelDate.impts,H, PanelDate.N_minHiRes, PanelDate.E_minHiRes)
    # calculate errors
      errors, ssq = estimate_homography_error(PanelDate, H,high_res = True)
      name_qual = 'HighRes'
      Extent = [E_axis.min(),E_axis.max(),N_axis.min(),N_axis.max()]
      Mapx = PanelDate.mappts[:,0]*10 # east
      Mapy = PanelDate.mappts[:,1]*10 # north
      #print('Mapxy',Mapx, Mapy )
  else:
    # calculate homography
      dst, H_low = rectify_low_res(PanelDate,img_rgb, plot = False)
      H = H_low
      E_max = PanelDate.E_max
      N_max = PanelDate.N_max
      minpts = PanelDate.minpts
      #print("minpts", minpts)
      E_min = minpts[0]
      N_min = minpts[1]
      pts = PanelDate.pts
      rows, cols, channels = dst.shape
      #print("dst.shape",dst.shape)
      E_axis = np.linspace(E_min, E_min + cols, cols)
      N_axis = np.linspace(N_min, N_min + rows, rows)
      Extent = [E_axis.min(),E_axis.max(),N_axis.min(),N_axis.max()]
      transE, transN = transform_points(PanelDate.impts,H, PanelDate.N_min, PanelDate.E_min)
    # calculate errors
      errors, ssq = estimate_homography_error(PanelDate, H,high_res = False)
      name_qual = 'LowRes'
      Mapx = PanelDate.mappts[:,0] # east
      Mapy = PanelDate.mappts[:,1] # north
      #print('Mapxy',Mapx, Mapy )



  Ptsx = pts[:,0]
  Ptsy = pts[:,1]
  TMapx= transE
  TMapy= transN
  EE = np.sqrt(errors[:,0]**2)
  EN = np.sqrt(errors[:,1]**2)
  EA = np.sqrt(EE * EN)
  Errors_all = np.column_stack((Mapx,Mapy,EA))
  xx, yy = np.meshgrid(np.arange(E_min, E_max,1),np.arange(N_min,N_max,1))
  intim_A = griddata((Errors_all[:,0],Errors_all[:,1]),Errors_all[:,2],(xx,yy))
  # make plot
  #print('Extent',Extent)
  filename = 'Error_surface_' + PanelDate.sitename + '_' + PanelDate.year + '_' + name_qual + '.PNG'
  fig = plt.figure(figsize = (16,6))
  fig.subplots_adjust(wspace = 0.3)
  # plot original image
  ax = plt.subplot(1,3,1)
  plt.title("Original image")
  plt.imshow(img_rgb)
  plt.scatter(x = PanelDate.impts[:,0], y = PanelDate.impts[:,1], c = 'r', s = 5,
              label = 'GCPs')
  # plot rectified image
  ax = plt.subplot(1,3,2)
  plt.title('Rectified image')
  plt.imshow(dst, extent = Extent, origin = 'lower')
  rw = plt.scatter(x = Ptsx + E_axis.min(), y = Ptsy + N_axis.min() ,c = 'r', s = 5,
              label = 'R-W points')
  rp = plt.scatter(x = TMapx, y = TMapy,c = 'b', s = 5,
              label = 'Reprojected pts')
  plt.legend((rw,rp),('Real-world coords','Reprojected coords'),loc='lower right',
           ncol=1,
           fontsize=8)
  # plot error surface
  ax = plt.subplot(1,3,3)
  plt.title("Error surface")
  plt.pcolormesh(xx,yy,intim_A)
  plt.axis([E_min, E_max,N_min,N_max])
  cbar = plt.colorbar( shrink = .5)
  plt.scatter(x= Mapx,y=Mapy, c = 'r', s = 15, label = 'R-W points')
  plt.scatter(x=TMapx, y=TMapy, c = 'b', s = 15, label = "Reprojected points")
  figure_name = "Rectification_error_surface_" + PanelDate.sitename + PanelDate.year +'.png'
  if save == True:
    plt.savefig(out_dir + os.sep + figure_name, bbox_inches = 'tight', dpi = 600)
    if plot == False:
        plt.close('all')
    else:
        plt.show()
        plt.close('all')
  else:
    if plot == False:
        plt.close('all')
    else:
        plt.show()
        plt.close('all')
    print('Figure not saved')

####### create PanelDate objects for each site

ref_im_path = os.getcwd() + os.sep + 'reference_images'
print("reference image path: ", ref_im_path)


# RC0220Ra

RC0220Ra_sitename = 'RC0220Ra'



# ---------------------------2017 survey ---------------------#
RC0220Ra_2017_panel_img_str = 'RC0220Ra_20171001_1341_und.JPG'
RC0220Ra_2017_panel_img = ref_im_path + os.sep + 'RC0220Ra_20171001_1341_und.JPG'
img_bgr = cv2.imread(RC0220Ra_2017_panel_img) # BGR image (cv2 default image type)
RC0220Ra_2017_img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB) # conver to RGB

RC0220Ra_2017_mappts = np.array([
#[227315.133,622494.517],
[227334.406,622484.373],
[227355.410,622490.895],
[227354.247,622469.411],
[227373.592,622498.210],
[227369.852,622509.492],
[227376.007,622540.577],
#[227374.201,622569.960],
[227351.439,622534.476],
#[227333.306,622534.074],
[227358.194,622518.934]
])

RC0220Ra_2017_impts = np.array([
#[1022.50,783.75],
[855.75 ,980.50],
[1058.25,1189.25],
[509.75 ,1211.75],
[1314.00,1288.00],
[1580.25,1230.25],
[2371.25,1232.00],
#[2979.50,1105.75],
[2017.75,988.75],
#[1888.00,798.25],
[1731.00,1122.50]
])

RC0220Ra_2017 = PanelDate(RC0220Ra_sitename, '2017',RC0220Ra_2017_impts,RC0220Ra_2017_mappts,RC0220Ra_2017_panel_img_str)

dst_RC0220Ra_2017, H_RC0220Ra_2017 = rectify_hi_res(RC0220Ra_2017,RC0220Ra_2017_img_rgb, plot = False)
dst_RC0220Ra_2017low, H_RC0220Ra_2017low = rectify_low_res(RC0220Ra_2017,RC0220Ra_2017_img_rgb, plot = False)
RC0220Ra_2017.load_homography_lowres(H_RC0220Ra_2017low)
RC0220Ra_2017.load_homography_hires(H_RC0220Ra_2017)
errors, ssq = estimate_homography_error(RC0220Ra_2017,H_RC0220Ra_2017)
print(f'Sum of Squared Errors for {RC0220Ra_2017.sitename} = {ssq}')
out = root_dir
plot_error_surface(RC0220Ra_2017,RC0220Ra_2017_img_rgb, Error_surface_dir, high_res = True, save = True)

################
# -----------------------2019 survey-----------------------
RC0220Ra_2019_panel_img_str = 'RC0220Ra_20191006_1414_und.JPG'
RC0220Ra_2019_panel_img = ref_im_path + os.sep + 'RC0220Ra_20191006_1414_und.JPG'
img_bgr = cv2.imread(RC0220Ra_2019_panel_img) # BGR image (cv2 default image type)
RC0220Ra_2019_img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB) # conver to RGB

RC0220Ra_2019_mappts = np.array([
[227383.838,622548.908],
[227372.365,622540.308],
[227378.031,622520.889],
[227395.177,622521.298],
[227376.936,622504.383],
[227357.183,622479.924],
[227348.993,622487.438],
#[227335.917,622493.066],
[227348.659,622510.260],
#[227363.098,622506.824],
[227356.289,622520.138],
#[227362.651,622534.235]
])

RC0220Ra_2019_impts = np.array([
[2458.50,1296.00],
[2139.75,1229.00],
[1741.00,1271.50],
[1910.25,1408.00],
[1310.00,1300.75],
[594.00,1248.25],
[768.50,1139.00],
#[868.50,1047.75],
[1302.00,1086.25],
#[1292.50,1161.00],
[1565.50,1126.25],
#[1922.00,1176.00]
])

RC0220Ra_2019 = PanelDate(RC0220Ra_sitename, '2019',RC0220Ra_2019_impts,RC0220Ra_2019_mappts,RC0220Ra_2019_panel_img_str)

dst_RC0220Ra_2019, H_RC0220Ra_2019 = rectify_hi_res(RC0220Ra_2019,RC0220Ra_2019_img_rgb, plot = False)
dst_RC0220Ra_2019low, H_RC0220Ra_2019low = rectify_low_res(RC0220Ra_2019,RC0220Ra_2019_img_rgb, plot = False)
RC0220Ra_2019.load_homography_lowres(H_RC0220Ra_2019low)
RC0220Ra_2019.load_homography_hires(H_RC0220Ra_2019)
errors, ssq = estimate_homography_error(RC0220Ra_2019,H_RC0220Ra_2019)
print(f'Sum of Squared Errors for {RC0220Ra_2019.sitename} = {ssq}')
out = root_dir
plot_error_surface(RC0220Ra_2019,RC0220Ra_2019_img_rgb, Error_surface_dir, high_res = True, save = True)

#### RC0307Rf

RC0307Rf_sitename = 'RC0307Rf'

#----------------------2017 survey -------------------------
RC0307Rf_2017_panel_img_str = 'RC0307Rf_20171002_1129_und.JPG'
RC0307Rf_2017_panel_img = ref_im_path + os.sep + 'RC0307Rf_20171002_1129_und.JPG'
img_bgr = cv2.imread(RC0307Rf_2017_panel_img) # BGR image (cv2 default image type)
RC0307Rf_2017_img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB) # conver to RGB

RC0307Rf_2017_mappts = np.array([
[219564.143,611772.081],
[219545.827,611789.288],
#[219541.506,611806.246],
#[219524.911,611831.131],
[219570.485,611856.435],
[219590.036,611862.600],
[219513.698,611772.908],
#[219540.082,611757.827],
[219517.774,611728.370],
[219513.184,611704.139]
])

RC0307Rf_2017_impts = np.array([
[1126.25,1290.00],
[1551.25,1094.75],
#[1879.00,984.00],
#[2330.50,496.50],
[2999.00,1158.75],
[3275.75,1353.25],
[1357.25,781.50],
#[995.25,1065.75],
[671.50,1003.25],
[349.50,1019.50]
])

RC0307Rf_2017 = PanelDate(RC0307Rf_sitename, '2017',RC0307Rf_2017_impts,RC0307Rf_2017_mappts,RC0307Rf_2017_panel_img_str)

dst_RC0307Rf_2017, H_RC0307Rf_2017 = rectify_hi_res(RC0307Rf_2017,RC0307Rf_2017_img_rgb, plot = False)
dst_RC0307Rf_2017low, H_RC0307Rf_2017low = rectify_low_res(RC0307Rf_2017,RC0307Rf_2017_img_rgb, plot = False)
RC0307Rf_2017.load_homography_lowres(H_RC0307Rf_2017low)
RC0307Rf_2017.load_homography_hires(H_RC0307Rf_2017)
errors, ssq = estimate_homography_error(RC0307Rf_2017,H_RC0307Rf_2017)
print(f'Sum of Squared Errors for {RC0307Rf_2017.sitename} = {ssq}')
out = root_dir
plot_error_surface(RC0307Rf_2017,RC0307Rf_2017_img_rgb, Error_surface_dir, high_res = True, save = True)

#----------------------2019 survey -------------------------

RC0307Rf_2019_panel_img_str = 'RC0307Rf_20191007_1145_und.JPG'
RC0307Rf_2019_panel_img = ref_im_path + os.sep + 'RC0307Rf_20191007_1145_und.JPG'
img_bgr = cv2.imread(RC0307Rf_2019_panel_img) # BGR image (cv2 default image type)
RC0307Rf_2019_img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB) # conver to RGB

RC0307Rf_2019_mappts = np.array([
[219542.644,611803.324],
#[219542.917,611790.645],
[219540.663,611778.078],
[219522.509,611775.796],
[219520.759,611764.155],
[219515.123,611754.640],
#[219529.343,611747.805],
[219515.438,611733.189],
#[219541.495,611757.829],
[219542.715,611735.480],
#[219549.039,611767.098],
[219571.880,611771.835]
])

RC0307Rf_2019_impts = np.array([
[1809.50,1095.75],
#[1577.75,1116.25],
[1354.50,1089.75],
[1388.75,968.00],
[1211.50,983.75],
[1093.75,968.50],
#[975.00,1085.00],
[782.00,1029.00],
#[1001.75,1129.25],
[641.50,1238.50],
#[1116.25,1174.75],
[1085.00,1426.00]
])

RC0307Rf_2019 = PanelDate(RC0307Rf_sitename, '2019',RC0307Rf_2019_impts,RC0307Rf_2019_mappts,RC0307Rf_2019_panel_img_str)

dst_RC0307Rf_2019, H_RC0307Rf_2019 = rectify_hi_res(RC0307Rf_2019,RC0307Rf_2019_img_rgb, plot = False)
dst_RC0307Rf_2019low, H_RC0307Rf_2019low = rectify_low_res(RC0307Rf_2019,RC0307Rf_2019_img_rgb, plot = False)
RC0307Rf_2019.load_homography_lowres(H_RC0307Rf_2019low)
RC0307Rf_2019.load_homography_hires(H_RC0307Rf_2019)
errors, ssq = estimate_homography_error(RC0307Rf_2019,H_RC0307Rf_2019)
print(f'Sum of Squared Errors for {RC0307Rf_2019.sitename} = {ssq}')
out = root_dir
plot_error_surface(RC0307Rf_2019,RC0307Rf_2019_img_rgb, Error_surface_dir, high_res = True, save = True)


#---------------------- RC1227R --------------------------------

RC1227R_sitename = 'RC1227R'

##---------------------2017----------------------------------------
RC1227R_2017_panel_img_str = 'RC1227R_20171010_1513_und.JPG'

RC1227R_2017_panel_img = ref_im_path + os.sep + RC1227R_2017_panel_img_str
img_bgr = cv2.imread(RC1227R_2017_panel_img) # BGR image (cv2 default image type)
RC1227R_2017_img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB) # conver to RGB

RC1227R_2017_mappts  = np.array([
[160099.193,581807.964],
[160136.131,581811.996],
#[160213.056,581821.401],
#[160254.185,581816.072],
[160254.034,581803.848],
#[160318.470,581774.720],
#[160202.135,581758.471],
[160256.535,581703.512],
[160280.922,581705.780]
])

RC1227R_2017_impts = np.array([
[555.50,1393.75],
[1035.00,1368.50],
#[2028.00,1279.75],
#[2568.75,1209.50],
[2582.50,1328.50],
#[3749.75,1312.75],
#[1746.00,1551.50],
[2874.75,1859.75],
[3522.00,1814.50]
])

RC1227R_2017 = PanelDate(RC1227R_sitename, '2017',RC1227R_2017_impts,RC1227R_2017_mappts, RC1227R_2017_panel_img_str)

dst_RC1227R_2017, H_RC1227R_2017 = rectify_hi_res(RC1227R_2017,RC1227R_2017_img_rgb, plot = False)
dst_RC1227R_2017low, H_RC1227R_2017low = rectify_low_res(RC1227R_2017,RC1227R_2017_img_rgb, plot = False)
RC1227R_2017.load_homography_lowres(H_RC1227R_2017low)
RC1227R_2017.load_homography_hires(H_RC1227R_2017)
errors, ssq = estimate_homography_error(RC1227R_2017,H_RC1227R_2017)
print(f'Sum of Squared Errors for {RC1227R_2017.sitename} = {ssq}')
out = root_dir
plot_error_surface(RC1227R_2017,RC1227R_2017_img_rgb, Error_surface_dir, high_res = True, save = True)

##---------------------2019----------------------------------------
RC1227R_2019_panel_img_str = 'RC1227R_20191015_1712_und.JPG'

RC1227R_2019_panel_img = ref_im_path + os.sep + RC1227R_2019_panel_img_str
img_bgr = cv2.imread(RC1227R_2019_panel_img) # BGR image (cv2 default image type)
RC1227R_2019_img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB) # conver to RGB

RC1227R_2019_mappts =  np.array([
[160267.562,581704.995],
[160256.714,581705.813],
[160260.376,581714.754],
[160212.118,581777.751],
#[160239.316,581755.533],
#[160195.581,581768.372],
[160184.070,581798.863],
#[160173.154,581796.720],
#[160167.604,581810.931],  # questionable
[160147.645,581810.787]
])

RC1227R_2019_impts = np.array([
[3158.50,1830.50],
[2866.50,1835.25],
[2915.00,1770.25],
[1948.25,1446.75],
#[2386.25,1555.00],
#[1666.00,1515.50],
[1596.75,1390.25],
#[1440.50,1423.75],
#[1420.25,1356.25], 	 # questionable
[1165.75,1371.00]
])

RC1227R_2019 = PanelDate(RC1227R_sitename, '2019', RC1227R_2019_impts, RC1227R_2019_mappts, RC1227R_2019_panel_img_str)

dst_RC1227R_2019, H_RC1227R_2019 = rectify_hi_res(RC1227R_2019,RC1227R_2019_img_rgb, plot = False)
dst_RC1227R_2019low, H_RC1227R_2019low = rectify_low_res(RC1227R_2019,RC1227R_2019_img_rgb, plot = False)
RC1227R_2019.load_homography_lowres(H_RC1227R_2019low)
RC1227R_2019.load_homography_hires(H_RC1227R_2019)
errors, ssq = estimate_homography_error(RC1227R_2017,H_RC1227R_2019)
print(f'Sum of Squared Errors for {RC1227R_2019.sitename} = {ssq}')
plot_error_surface(RC1227R_2019,RC1227R_2019_img_rgb, Error_surface_dir, high_res = True, save = True)



#-----------------------------------RC1459L----------------------------------------
RC1459L_sitename = 'RC1459L'

##------------------------------------2017------------------------------
RC1459L_2017_panel_img_str = 'RC1459L_20171012_1509_und.JPG'

RC1459L_2017_panel_img = ref_im_path + os.sep + RC1459L_2017_panel_img_str
img_bgr = cv2.imread(RC1459L_2017_panel_img) # BGR image (cv2 default image type)
RC1459L_2017_img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB) # conver to RGB

RC1459L_2017_mappts = np.array([
[147592.391,596329.092],
[147596.473,596322.600],
#[147598.165,596303.590],
[147604.296,596273.391],
[147601.765,596263.622],
[147612.868,596274.317],
#[147616.184,596290.376],
#[147612.392,596303.143],
[147616.778,596313.255]
])

RC1459L_2017_impts = np.array([
[570.50,1850.00],
[638.75,1769.00],
#[1046.75,1571.25],
[1442.50,1334.25],
[1641.00,1272.50],
[1221.75,1222.50],
#[876.50,1377.25],
#[695.75,1476.25],
[339.75,1471.50]
])

RC1459L_2017 = PanelDate(RC1459L_sitename, '2017', RC1459L_2017_impts, RC1459L_2017_mappts, RC1459L_2017_panel_img_str)

dst_RC1459L_2017, H_RC1459L_2017 = rectify_hi_res(RC1459L_2017,RC1459L_2017_img_rgb, plot = False)
dst_RC1459L_2017low, H_RC1459L_2017low = rectify_low_res(RC1459L_2017,RC1459L_2017_img_rgb, plot = False)
RC1459L_2017.load_homography_lowres(H_RC1459L_2017low)
RC1459L_2017.load_homography_hires(H_RC1459L_2017)
errors, ssq = estimate_homography_error(RC1459L_2017,H_RC1459L_2017)
print(f'Sum of Squared Errors for {RC1459L_2017.sitename} = {ssq}')
out = root_dir
plot_error_surface(RC1459L_2017,RC1459L_2017_img_rgb, Error_surface_dir, high_res = True, save = True)

##------------------------------------2019--------------------
RC1459L_2019_panel_img_str = 'RC1459L_20191017_1242_und.JPG'

RC1459L_2019_panel_img = ref_im_path + os.sep + RC1459L_2019_panel_img_str
img_bgr = cv2.imread(RC1459L_2019_panel_img) # BGR image (cv2 default image type)
RC1459L_2019_img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB) # conver to RGB

RC1459L_2019_mappts = np.array([
[147595.349,596313.802],
[147593.378,596324.128],
[147612.757,596308.259],
#[147604.226,596300.471],
[147603.190,596279.127],
[147608.263,596274.265],
#[147602.937,596293.985],
#[147611.350,596293.991],
#[147615.983,596298.046]
])

RC1459L_2019_impts = np.array([
[896.25,1609.50],
[690.75,1762.25],
[574.00,1478.50],
#[953.75,1405.75],
[1385.75,1348.00],
[1218.50,1276.75],
#[1120.50,1387.25],
#[915.75,1349.75],
#[724.75,1380.75]
])

RC1459L_2019 = PanelDate(RC1459L_sitename, '2019', RC1459L_2019_impts, RC1459L_2019_mappts, RC1459L_2019_panel_img_str)

dst_RC1459L_2019, H_RC1459L_2019 = rectify_hi_res(RC1459L_2019,RC1459L_2019_img_rgb, plot = False)
dst_RC1459L_2019low, H_RC1459L_2019low = rectify_low_res(RC1459L_2019,RC1459L_2019_img_rgb, plot = False)
RC1459L_2019.load_homography_lowres(H_RC1459L_2019low)
RC1459L_2019.load_homography_hires(H_RC1459L_2019)
errors, ssq = estimate_homography_error(RC1459L_2017,H_RC1459L_2019)
print(f'Sum of Squared Errors for {RC1459L_2019.sitename} = {ssq}')
out = root_dir
plot_error_surface(RC1459L_2019,RC1459L_2019_img_rgb, Error_surface_dir, high_res = True, save = True)

PanelDate_dict = {'RC0220Ra.2017': RC0220Ra_2017,
                    'RC0220Ra.2019': RC0220Ra_2019,
                    'RC0307Rf.2017': RC0307Rf_2017,
                    'RC0307Rf.2019': RC0307Rf_2019,
                    'RC1227R.2017': RC1227R_2017,
                    'RC1227R.2019': RC1227R_2019,
                    'RC1459L.2017': RC1459L_2017,
                    'RC1459L.2019': RC1459L_2019}
