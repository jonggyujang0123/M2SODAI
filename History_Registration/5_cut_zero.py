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

img_path = path + '/val/'
target_path = path + '/val/'

# Size Configuration

if not os.path.exists(target_path):
    os.makedirs(target_path)
    
file_list = sorted(glob.glob(img_path + '*[!_hsi].jpg'))

for i, img_name in enumerate(tqdm(file_list)): 
    # if i < 18:
    #     continue
    org_idx = os.path.splitext(os.path.basename(img_name))[0] 
    tar_idx = org_idx

    img = cv2.imread(img_name)
    # img[img==255] = 0
    hsi = sio.loadmat(img_name.replace('jpg', 'mat'))['data']
    hsi_view = HSItensor2imgs(hsi)
    ndvi_map = ndvi(hsi + 1e-13,25,72)
    ndvi_map = np.uint8(255* ndvi_map)


    img[(cv2.resize(hsi_view.mean(axis=2), (1600,1600))==0),:]=0
    hsi[(cv2.resize(img.mean(axis=2), (224,224))==0),:]=0
    hsi_view[(cv2.resize(img.mean(axis=2), (224,224))==0),:]=0
    ndvi_map[(cv2.resize(img.mean(axis=2), (224,224))==0)]=0
    
    


    # plt.imshow(np.concatenate((img, cv2.resize(hsi_view, (1600,1600)))))
    
    # plt.show()
    
    
    new_img_name = target_path + tar_idx + '.jpg'
    new_hsi_name = target_path + tar_idx + '.mat'
    new_hsi_jpg_name = target_path + tar_idx + '_hsi.jpg'
    new_ndvi_jpg_name = target_path + tar_idx + '_ndvi.jpg'
    # input(f"{i}-th data, press enter to continue")

    cv2.imwrite(new_img_name, img)
    sio.savemat(new_hsi_name, {'data':hsi})
    cv2.imwrite(new_hsi_jpg_name, hsi_view)
    cv2.imwrite(new_ndvi_jpg_name, ndvi_map)
