'''
Generate a random stratified sample of imagery from a directory
    inputs:
    directory - provide a path to a directory containing imagery
    sample_size - provide the size of the sample
    returns:
    this script creates a directory called Sample_sizexx containing the sample of imagery
'''
# imports

#!pip install datetimerange
from sklearn.model_selection import train_test_split
import os
import datetime as dt
import pandas as pd
from datetimerange import DateTimeRange
import shutil

# functions
def get_datetime(image_filename):
  '''
  image_filename = str(filename for an image)
  ouputs: ext, date, time, dt_obj
  ext = str(extention)
  date = str(date)
  time = str(time)
  dt_obj = datetime object
  '''
  ext = os.path.splitext(image_filename)[1][1:]
  #print(ext, " --ext")
  date = image_filename.split(os.sep)[-1].split('RC')[-1].split('_')[1]
  #print(date, " --date")
  time = image_filename.split(os.sep)[-1].split('RC')[-1].split('_')[2].split('.'+ext)[0]
  #print(time, " --time")
  dt_obj = dt.datetime.strptime(date + '_' +time, '%Y%m%d_%H%M')
  #print(dt_obj, " --dt_obj")
  return ext, date, time, dt_obj

def create_imagedate_df(im_list):
     '''
     requires:
     >import datetime as dt
     >import os
     >import pandas as pd
     >get_datetime()
     input: list of image filenames
     outputs: a dataframe containing the following columns:
      'Filename' = str(filename)
      'Date' = str(date) : YYYYMMDD
      'Month' = int from 1 to 12
      'Datetime' = datetime object
     '''
     filenames = []
     dates = []
     months = []
     datetimes = []
     for file in im_list:
       e, d, t, dt_obj = get_datetime(file)
       filenames.append(file)
       dates.append(d)
       months.append(dt_obj.month)
       datetimes.append(dt_obj)
       data = {'Filename':filenames,'Date':dates, 'Month': months, 'Datetime':datetimes}
       df = pd.DataFrame(data, columns = data.keys())
     return df

def test_if_HFE(date_time, HFE_dict):
    '''
    this function tests if a specific datetime took place within an HFE
    inputs:
    date_time = a date_time object YYYYmmdd_HHMM
    HFE_dict = a dict containing HFEs and DateTimeRange objects
        {HFE2012:DateTimeRange,...,}
    outputs:
        True = if date_time in DateTimeRanges of any of the HFEs in HFE_dict
        False = if not date_time in DateTimeRanges of any of the HFEs in HFE_dict
    '''
    HFE_list = HFE_dict.keys()
    values = []
    for hfes in HFE_list:
        dt_range = HFE_dict[hfes]
    if date_time in dt_range:
      values.append(1)
    else:
      values.append(0)
    if sum(values) == 1:
       return True
    else:
       return False

def convert_Month_to_13_if_HFE(df, HFE_dict):
    '''This function converts the 'Month' to 13 if the image occurs during an HFE in HFE_dict
    inputs:
        df
        HFE_dict
    returns:
        new df with months of HFE dates transformed to = 13
    '''
    for i,j in df.iterrows():
        if test_if_HFE(j['Datetime'],HFE_dict):
            j['Month'] = 13
    return df

#
try_again = True
while try_again == True:
  images_path_in = input("Please enter the path to the directory containing images to be sampled:")
  # check if provided dir is valid
  print(f"You provided the following directory: {images_path_in}")
  if os.path.exists(images_path_in):
    print("Valid Filepath")
    image_list = []
    for file in os.listdir(images_path_in):
      if file.endswith('.JPG') or file.endswith('.jpg'):
        image_list.append(file)
      else:
        continue
    try_again = False
    print("image_list:\n",image_list)
  else:
    print("path provided is either invalid or does not exist!")


print(f'You provided the following path:{images_path_in}')
print(f'It contains {len(os.listdir(images_path_in))} files')

try_again = True
while try_again == True:
    try:
        sample_size = int(input("How large of a sample would you like? (int):"))
        try_again = False
    except:
        print('ERROR invalid input, provide an interger')
    #else:
    #    print('ERROR invalid input, provide an interger')


print(f'You chose a sample_size of:{sample_size}')

# hard coded variables
HFE2012a= dt.datetime.strptime('20121119_1200', '%Y%m%d_%H%M')
HFE2012b= dt.datetime.strptime('20121125_2359', '%Y%m%d_%H%M')
HFE2013a= dt.datetime.strptime('20131111_0900', '%Y%m%d_%H%M')
HFE2013b= dt.datetime.strptime('20131117_2359', '%Y%m%d_%H%M')
HFE2014a= dt.datetime.strptime('20141110_0900', '%Y%m%d_%H%M')
HFE2014b= dt.datetime.strptime('20141116_2359', '%Y%m%d_%H%M')
HFE2016a= dt.datetime.strptime('20161107_0600', '%Y%m%d_%H%M')
HFE2016b= dt.datetime.strptime('20161112_2359', '%Y%m%d_%H%M')
HFE2018a= dt.datetime.strptime('20181105_0600', '%Y%m%d_%H%M')
HFE2018b= dt.datetime.strptime('20181111_2359', '%Y%m%d_%H%M')

HFEs = {'HFE2012':DateTimeRange(HFE2012a,HFE2012b),
        'HFE2013':DateTimeRange(HFE2013a,HFE2013b),
        'HFE2014':DateTimeRange(HFE2014a,HFE2014b),
        'HFE2016':DateTimeRange(HFE2016a,HFE2016b),
        'HFE2018':DateTimeRange(HFE2018a,HFE2018b)}

# script

print(image_list)
print("Creating DataFrame, please wait....")
df = create_imagedate_df(image_list)
new_df = convert_Month_to_13_if_HFE(df,HFEs)
print("Searching for HFE images, please wait....")

stratified_sample, _ = train_test_split(new_df,train_size = sample_size, stratify = new_df['Month'])
print('Random stratified sample generated...')
stratified_sample

out_dir = os.path.join(images_path_in + os.sep + 'Sample_size' + str(sample_size))
if not os.path.exists(out_dir):
  os.makedirs(out_dir)
  print(f'creating output directory {out_dir}')

file_num = 1
for i,j in stratified_sample.iterrows():
  file = os.path.join(images_path_in + os.sep + j['Filename'])
  shutil.copy(file,out_dir)
  print(f'Copying file {file_num} of {sample_size}... ')
  file_num += 1
