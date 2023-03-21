# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 07:06:23 2023

@author: h_min
"""

# Load requied library 
import numpy as np
import matplotlib.pyplot as plt
import tqdm
import re

from langdetect import detect, DetectorFactory
DetectorFactory.seed = 0

# Set the file path
file_path = "C:/Users/h_min/repos/cs234-dreamchess/decision-transformer/dataset/"

# Load the data
MASK_CHAR = u"\u2047"
full_data = open(file_path + 'all_with_filtered_anotations_since1998.txt').readlines()

# Check the length of the data 
len(full_data)