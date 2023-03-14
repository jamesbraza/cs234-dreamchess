# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 07:16:11 2023

@author: h_min
"""

echo "Checking Kaggle Install..."
if ! command -v kaggle &> /dev/null
then
	echo "Kaggle is not found"
	pip3 install --user kaggle
	pip install --user kaggle
	pip3 install kaggle
	pip install kaggle
else
	echo "Kaggle is installed"
fi

echo "Checking Kaggle Dataset..."

if [ -e "./data/datasets/35-million-chess-games.zip" ]
then
	echo "Kaggle Dataset Downloaded"
else
	cd data/datasets
	kaggle datasets download milesh1/35-million-chess-games
	cd ../..
fi

if [ -e "./decision-transformer/dataset/all_with_filtered_anotations_since1998.txt" ]
then
	echo "Kaggle Dataset Unzipped"
else
	unzip ./decision-transformer/dataset/35-million-chess-games.zip -d ./decision-transformer/dataset/
fi