#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import numpy as np
import cv2
import scipy.io as sio
import glob
from itertools import product
from matplotlib import pyplot as plt
import argparse
from PIL import Image, ImageEnhance
import spectral.io.envi as envi
from spectral import *
from tqdm import tqdm 
import shutil
import json

parser = argparse.ArgumentParser(description = 'Options')
parser.add_argument('--dir', help = 'directory of dataset')
parser.add_argument('--start', type=int, help = 'starting index')
args = parser.parse_args()

path = args.dir
start = args.start
img_path = path + '/data_save_coincided_2/'
target_path = path + '/data_save_numbered/'

# Size Configuration

if not os.path.exists(target_path):
    os.makedirs(target_path)
    
file_list = sorted(glob.glob(img_path + '*[!_hsi][!_ndvi].jpg'))

for i, img_name in enumerate(tqdm(file_list)): 
    if os.path.exists(img_name.replace('.jpg','.json')):
        shutil.copy(img_name, f'{target_path}{start+i}.jpg')
        shutil.copy(img_name.replace('.jpg','.mat'), f'{target_path}{start+i}.mat')
        shutil.copy(img_name.replace('.jpg','.json'), f'{target_path}{start+i}.json')
        json_path = f'{target_path}{start+i}.json'
        with open(json_path, "r") as json_file:
            json_data = json.load(json_file)
            json_data["imagePath"] = f'{start+i}.jpg'
        with open(json_path, 'w') as outfile:
            json.dump(json_data, outfile) 
    
