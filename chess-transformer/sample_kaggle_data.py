# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 07:06:23 2023

@author: h_min
"""


import random
import pandas as pd

file_path = "C:/Users/h_min/repos/cs234-dreamchess/chess-transformer/dataset/"
full_data = open(file_path + 'processed_kaggle_result_elo.txt').readlines()

# Lists:
gameslist = []

#Placed In List:
for line in full_data:
        gameslist.append(line.replace("\n", "|"))

#full_data.close()

rand_options = gameslist  # duplicate list, better than editting the input list
rand_vals = []
length = 500  # configure to 10, or how ever many random lines you want

for _ in range(length):
    rand_val = random.choice(rand_options)
    rand_vals.append(rand_val)    
    rand_options.remove(rand_val)  # remove from list so you don't get duplicates (unless you don't mind those)

train_data_sampled = "".join(rand_vals).replace("|","\n")

f = open(file_path + "kaggle_train_data_sampled_500.txt", "w+")
f.writelines(train_data_sampled)
f.close()

