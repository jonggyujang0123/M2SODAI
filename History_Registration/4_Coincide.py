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

def enhancer(img):
    PIL_img = Image.fromarray(img)
    out = ImageEnhance.Brightness(PIL_img).enhance(1.6)
    #out = ImageEnhance.Contrast(out).enhance(0.6)
    #out = ImageEnhance.Color(out).enhance(0.2)
    out = ImageEnhance.Sharpness(out).enhance(2.2)
    out = np.array(out)
    return out

def HSItensor2imgs(img):
    img = (255*img/ 0.255).astype(np.uint8)
    idx = [17,34,53]
    return enhancer(img[:,:,idx])


parser = argparse.ArgumentParser(description = 'Options')
parser.add_argument('--dir', help = 'directory of dataset')
args = parser.parse_args()

path = args.dir

img_path = path + '/data_save_selected/'
target_path = path + '/data_save_coincided/'

# Size Configuration

if not os.path.exists(target_path):
    os.makedirs(target_path)
    
file_list = sorted(glob.glob(img_path + '*[!_hsi].jpg'))


# file_list = ('./data/test/30.jpg',)

def img_view(dmc, hsi):
    a=0.2
    while(a <= 0.8):
        b = 1.0 - a
        dst = cv2.addWeighted(dmc[:,:,2::-1], a, hsi, b, 0)
        plt.imshow(dst)
        plt.show()
        a += 0.2

for i, img_name in enumerate(file_list): 
    if i < 273:
        continue

    input(f"{i}/{len(file_list)}-th data, Press Enter to continue...")
    org_idx = os.path.splitext(os.path.basename(img_name))[0] 
    tar_idx = org_idx
    img = cv2.resize(cv2.imread(img_name), (1600,1600))
    # img[img==255] = 0
    hsi = sio.loadmat(img_name.replace('jpg', 'mat'))['data']
    for k in range(999):
        if k>0:
            input_x=  input(f"{i}/{len(file_list)}-th data, offset (ex. 100 100), if satisfied enter y:")
            if input_x in ['y', 'n']:
                break
            offset = [int(x) for x in input_x.split()][::-1]
        else:
            offset = (0,0)
        if not len(offset)==2:
            offset = (0,0)
        
        hsi_target = np.zeros_like(hsi)
        hsi_target[max(0,offset[0]): offset[0] + hsi.shape[0], max(-offset[1],0): -offset[1] + hsi.shape[1],:] = hsi[max(0,-offset[0]): - offset[0] + hsi.shape[0], max(0,offset[1]): + offset[1] + hsi.shape[1],:].copy()

        hsi_view = cv2.resize(HSItensor2imgs(hsi_target), (1600,1600))
        img_view(img,hsi_view)
    if input_x== 'n':
        continue
    plt.show()
    img[cv2.resize(hsi_target, (1600,1600)).mean(axis=2)==0,:]=0
    plt.imshow(np.concatenate((img, hsi_view)))
    ndvi_map = ndvi(hsi_target + 1e-5,50,86)
    ndvi_map = np.floor(255* ndvi_map / np.max(ndvi_map)).astype(int)
    new_img_name = target_path + tar_idx + '.jpg'
    new_hsi_name = target_path + tar_idx + '.mat'
    new_hsi_jpg_name = target_path + tar_idx + '_hsi.jpg'
    new_ndvi_jpg_name = target_path + tar_idx + '_ndvi.jpg'
    cv2.imwrite(new_img_name, img)
    sio.savemat(new_hsi_name, {'data':hsi_target})
    cv2.imwrite(new_hsi_jpg_name, HSItensor2imgs(hsi_target))
    cv2.imwrite(new_ndvi_jpg_name, ndvi_map)
