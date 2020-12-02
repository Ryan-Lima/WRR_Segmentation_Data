#!/usr/bin/env python3

# -*- coding: utf-8 -*-
"""
Created November, 2020
@author: mkf58
"""

import os
import sys
import glob
from datetime import datetime

def get_images(start_date, end_date, filedir = '.'):
    # Start by getting all files
    outfiles = []
    allfiles = os.walk(filedir)
    
    # Loop through all filepaths and get all jpeg files
    for root, dirs, files in allfiles:
        for name in files:
            outtemp = os.path.join(root,name)
            # Check that the file is a JPG or jpg file
            if (outtemp.find('.JPG') != -1) | (outtemp.find('.jpg') != -1):
                outfiles.append(outtemp)
    
    # Find files in the correct date range
    outjpg = []
    for file in outfiles:
        date_str = file.split('_')[len(file.split('_'))-2]
        time_str = file.split('_')[len(file.split('_'))-1].split('.')[0]
        date_time_str = date_str + time_str
        try:
            if (datetime.strptime(date_time_str, '%Y%m%d%H%M') >= datetime.strptime(start_date, '%Y-%m-%d')) & (datetime.strptime(date_time_str, '%Y%m%d%H%M') <= datetime.strptime(end_date, '%Y-%m-%d')):
                outjpg.append(file)
        except ValueError:
            print(file, ", ", date_time_str, ": is not a proper date value %Y%m%d%H%M")
    
    # Return a list of jpeg files
    return(outjpg)    