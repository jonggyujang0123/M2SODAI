# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 23:07:30 2022

@author: jgjang
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import scipy.io as sio
from tqdm import tqdm
import cv2
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image, ImageEnhance
import itertools

def enhancer(img):
    PIL_img = Image.fromarray(img)
    out = ImageEnhance.Brightness(PIL_img).enhance(1.5)
    out = ImageEnhance.Contrast(out).enhance(0.8)
    out = ImageEnhance.Color(out).enhance(0.8)
    out = ImageEnhance.Sharpness(out).enhance(2.5)
    out = np.array(out)
    return out

def HSItensor2imgs(img):
    img = (255*img/ 0.235).astype(np.uint8)
    idx = [17,34,53]
    return enhancer(img[:,:,idx])

def HSItensor2imgs_chan(img, idx):
    img = (255*img/ 0.235).astype(np.uint8)
    return enhancer(img[:,:,idx])

data_ind = 263
hsi_mat_name = f'../data/test/{data_ind}.mat'
jpg_name = f'../data/test/{data_ind}.jpg'

hsi = sio.loadmat(hsi_mat_name)['data'] # (224, 224, 127)
jpg = cv2.imread(jpg_name) # (1600, 1600, 3)

jpg_hsi = HSItensor2imgs(hsi)[:,:,2::-1]


plt.imshow(jpg_hsi)
plt.show()

plt.imshow(jpg)
plt.show()

'''
HSIs
obj1 x: 47-49, y: 88-89 
obj2 x: 95-98, y: 100-101
obj3 x: 64-66  y: 151
obj4 x: 70-72  y: 152

JPGs
obj1 x: 357-364   y : 640-646
obj2 x: 688-698   y: 712-720
obj3 x: 1035-1042 y: 617-624
obj4 x: 490-498   y: 985-995
'''

# Pixels
pixels_start_x_hsi = [150, 150, 130, 83, 78, 161]
pixels_end_x_hsi =   [153, 153, 133, 86, 81, 164]
pixels_start_y_hsi = [15, 78, 154, 191, 195, 80]
pixels_end_y_hsi =   [18, 81, 157, 193, 197, 82]

pixels_start_x_jpg = [1060, 1072, 944, 1532, 1270, 978]
pixels_end_x_jpg =   [1074, 1084, 954, 1548, 1286, 988]
pixels_start_y_jpg = [105, 558, 1110, 1293, 1308, 1375]
pixels_end_y_jpg =   [122, 572, 1128, 1308, 1316, 1384]

#Plot patches
## HSI
for i in range(len(pixels_start_x_hsi)):
    plt.imshow(jpg_hsi[
        pixels_start_y_hsi[i]:pixels_end_y_hsi[i],
        pixels_start_x_hsi[i]:pixels_end_x_hsi[i],
        :
        ])
    plt.show()
## JPG

for i in range(len(pixels_start_x_jpg)):
    plt.imshow(jpg[
        pixels_start_y_jpg[i]:pixels_end_y_jpg[i],
        pixels_start_x_jpg[i]:pixels_end_x_jpg[i],
        :
        ])
    plt.show()
    cv2.imwrite(f'jpg_object_{i}.png', jpg[
        pixels_start_y_jpg[i]:pixels_end_y_jpg[i],
        pixels_start_x_jpg[i]:pixels_end_x_jpg[i],
        2::-1
        ])



