# Parameters to remove image distortion for 10 sites
# created 12/1/2020 by Ryan Lima

import numpy as np

site_calibration_list = ["RC0220Ra","RC0235L","RC0307Rf","RC0566R","RC0917R","RC1044R", "RC1227R","RC1377L","RC1459L","RC1946L"]

class SiteDistortion:
  '''
  A SiteDistortion object contains the information required to remove distortion
   from imagery from a site.Removing distortion inherent in the camera-lens system
   is a requisite for further processing.

  Attributes:
  '''
  def __init__(self, sitename, A_Mat, k1, k2, p1 = 0, p2 = 0, k3 = 0, cal_year = '2018'):
    self.sitename = sitename
    if sitename not in site_calibration_list:
      print('ERROR, provided sitename not in list')
    self.cal_year = cal_year
    cal_year_list = ['2018']
    if cal_year not in cal_year_list:
      print('ERROR, provided calibration year not in calibtartion year list')
    self.A_Mat = A_Mat
    self.A_MatT = A_Mat.T
    self.k1 = k1
    self.k2 = k2
    self.p1 = p1
    self.p2 = p2
    self.k3 = k3
    self.distCoeff = np.array([k1,k2,p1,p2,k3])

  def __str__(self):
    return 'Site Distortion Objects for sandbar site: ' + self.sitename + ' : Calibration from year- ' + self.cal_year

RC0220Ra_A_Mat = np.array([[3876.05658615236,0,0],[0,3895.72930299343,0],[1913.44438724644,1371.72465447568,1]])
RC0220Ra_k1 = -0.103855357514449
RC0220Ra_k2 = 0.0917758599801266

RC0235L_A_Mat = np.array([[9642.56936556379,0,0],[0,9702.96972911076,0],[2405.60306964555,1649.94706091545,1]])
RC0235L_k1 = -0.147146049874411
RC0235L_k2 = 2.08662376048151

RC0307Rf_A_Mat = np.array([[3634.13529494755,0,0],[0,3651.71844146905,0],[1842.76994991078,1240.98226007733,1]])
RC0307Rf_k1 = -0.134888310194568
RC0307Rf_k2 = 0.0987327446646334

RC0566R_A_Mat =  np.array([[5869.33400908270,0,0],[0,5899.93215249742,0],[2656.94098419770,1842.68353050738,1]])
RC0566R_k1 = -0.128073490110434
RC0566R_k2 = 0.225794251252751

RC0917R_A_Mat = np.array([[3559.25556518922,0,0],[0,3578.64948898251,0],[2089.07486712043,1489.67863631963,1]])
RC0917R_k1 = -0.201246396977593
RC0917R_k2 = 0.217641611195368

RC1044R_A_Mat = np.array([[5577.47839442312,0,0],[0,5584.03316519538,0],[1814.35855674101,1343.42667612623,1]])
RC1044R_k1 = -0.0197005073114554
RC1044R_k2 = -0.293851677913236

RC1227R_A_Mat = np.array([[3214.03657558362,0,0],[0,3237.12022154347,0],[1994.65565535286,1339.21266495130,1]])
RC1227R_k1 = -0.163614941137452
RC1227R_k2 = 0.0917298853261050

RC1377L_A_Mat = np.array([[4273.86402116652,0,0],[0,4248.58867168531,0],[2593.76355729159,1839.01078637633,1]])
RC1377L_k1 = -0.207253410437886
RC1377L_k2 = 0.184520995533379

RC1459L_A_Mat = np.array([[4503.43628071119,0,0],[0,4503.22800203952,0],[1903.19528000849,1338.57063922250,1]])
RC1459L_k1 = -0.123557348919849
RC1459L_k2 = 0.322587494511831

RC1946L_A_Mat = np.array([[5937.12030730829,0,0],[0,5928.66162145540,0],[1972.83678746356,1344.51130810201,1]])
RC1946L_k1 = -0.0857133587372006
RC1946L_k2 = 0.291761801160749


RC0220Ra_Undistort = SiteDistortion('RC0220Ra', RC0220Ra_A_Mat, RC0220Ra_k1, RC0220Ra_k2)
RC0235L_Undistort = SiteDistortion('RC0235L',RC0235L_A_Mat, RC0235L_k1, RC0235L_k2)
RC0307Rf_Undistort = SiteDistortion('RC0307Rf', RC0307Rf_A_Mat, RC0307Rf_k1, RC0307Rf_k2)
RC0566R_Undistort = SiteDistortion('RC0566R', RC0566R_A_Mat, RC0566R_k1, RC0566R_k2)
RC0917R_Undistort = SiteDistortion('RC0917R', RC0917R_A_Mat, RC0917R_k1, RC0917R_k2)
RC1044R_Undistort = SiteDistortion('RC1044R', RC1044R_A_Mat, RC1044R_k1, RC1044R_k2)
RC1227R_Undistort = SiteDistortion('RC1227R', RC1227R_A_Mat, RC1227R_k1, RC1227R_k2)
RC1377L_Undistort = SiteDistortion('RC1377L',RC1377L_A_Mat,RC1377L_k1, RC1377L_k2)
RC1459L_Undistort = SiteDistortion('RC1459L', RC1459L_A_Mat, RC1459L_k1, RC1459L_k2)
RC1946L_Undistort = SiteDistortion('RC1946L', RC1946L_A_Mat, RC1946L_k1, RC1946L_k2)

site_calibration_dict = {"RC0220Ra": RC0220Ra_Undistort,
                        "RC0235L": RC0235L_Undistort,
                        "RC0307Rf": RC0307Rf_Undistort,
                        "RC0566R": RC0566R_Undistort,
                        "RC0917R": RC0917R_Undistort,
                        "RC1044R": RC1044R_Undistort,
                         "RC1227R": RC1227R_Undistort,
                         "RC1377L": RC1377L_Undistort,
                         "RC1459L": RC1459L_Undistort,
                         "RC1946L":RC1946L_Undistort}
