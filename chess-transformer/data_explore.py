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
file_path = "C:/Users/h_min/repos/cs234-dreamchess/chess-transformer/dataset/"

# Load the data
MASK_CHAR = u"\u2047"
full_data = open(file_path + 'processed_kaggle_result_elo.txt').readlines()

# Check the length of the data 
len(full_data)

s_data = full_data[1]

file2 = open(r"D:\Text\MyFile2.txt","w+")

s_data.to_csv("s_kaggle2.txt")


with open('C:/Users/h_min/repos/cs234-dreamchess/chess-transformer/dataset/s_kaggle2.txt', 'w') as outfile:
    outfile.write('\n'.join(str(i) for i in s_data))