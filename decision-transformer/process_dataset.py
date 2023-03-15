# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 07:18:56 2023

@author: h_min
"""

import tqdm
import matplotlib.pyplot as plt
import numpy as np
import os
#import random
import re
import questionary
from art import tprint

max_game_length = 512
min_game_length = 20

def preprocess_kaggle():
    """
    A funciton to pre-process and clean Kaggle chess play data
    Input: Original 3.5 million Kaggle Chess play data
    Output: Cleaned Kaggle Chess play data
    """
    
    print("Now preprocessing all_with_filtered_anotations_since1998.txt")

    write_folder = "./dataset/dataset-cleaned/"
    if not os.path.exists(write_folder):
        os.makedirs(write_folder)

    # check if this file has already been preprocessed
    if os.path.exists("./dataset/dataset-cleaned/kaggle_cleaned.txt"):
        response = questionary.confirm("It appears that the kaggle file has already been preprocessed; reprocess?").ask()
        if not response:
            return
        os.remove("./dataset/dataset-cleaned/kaggle_cleaned.txt")

    unprocessed_kaggle_lines = open("./dataset/all_with_filtered_anotations_since1998.txt", "r").readlines()[5:]

    processed_kaggle_lines = open("./dataset/dataset-cleaned/kaggle_cleaned.txt", "w")

    line_length = []
    for line in tqdm.tqdm(unprocessed_kaggle_lines):
        split_line = line.split()
        for index, token in enumerate(split_line):
            if index % 2 == 0:
                split_line[index] = token[3:]
            else:
                split_line[index] = token[1:]
        output_line = " ".join(split_line[17:]) + "\n"
        if output_line == "\n":
            continue
        output_line = re.sub(r'[0-9]*\.', '', output_line)
        if len(output_line) <= max_game_length and len(output_line) >= min_game_length and '[' not in output_line and ']' not in output_line:
            processed_kaggle_lines.writelines(output_line)
            line_length.append(len(output_line))

    x = np.array(line_length)

    plt.hist(x, density=True, bins=100)  # density=False would make counts
    plt.ylabel('Relative Frequency')
    plt.xlabel('Sequence Length')
    plt.show()


def main():
    tprint("Decision Transformer Preprocessor")
    preprocess_kaggle()

if __name__ == '__main__':
    main()