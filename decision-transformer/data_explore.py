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

import chess.pgn

from langdetect import detect, DetectorFactory
DetectorFactory.seed = 0

# Set the file path
file_path = "C:/Users/h_min/repos/cs234-dreamchess/decision-transformer/dataset/"
fname = "ficsgamesdb_2021_standard2000_nomovetimes_280587.pgn"
# Load the data
MASK_CHAR = u"\u2047"
full_data = open(file_path + 'all_with_filtered_anotations_since1998.txt').readlines()

# Check the length of the data 
len(full_data)


# Set the file path
file_path2 = "C:/Users/h_min/repos/cs234-dreamchess/decision-transformer/dataset/dataset-cleaned/"

# Load the data
MASK_CHAR = u"\u2047"
full_data2 = open(file_path2 + 'kaggle_cleaned.txt').readlines()
