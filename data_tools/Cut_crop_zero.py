#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import numpy as np
import cv2
import scipy.io as sio
import glob
from itertools import product
from matplotlib import pyplot as plt
from data_tools.Image_HSI_to_JPG import HSItensor2imgs

img_path = './data/test/'
target_path = './data/test_new/'

# Size Configuration

jpg_size = (640,640)
hsi_size = (90,90)
x_grid = 3
y_grid = 3
# jpg_grd = (np.arange(0, 1601, 1600/2)-np.arange(0, 641, 640/2)).astype(int)
# hsi_grd = (np.arange(0, 225, 224/2)-np.arange(0, 91, 90/2)).astype(int)
if not os.path.exists(target_path):
    os.makedirs(target_path)
    
# file_list = glob.glob(img_path + '*.jpg')


file_list = ('./data/test/30.jpg',)

def img_view(dmc, hsi):
    a=0.2
    while(a <= 0.8):
        b = 1.0 - a
        dst = cv2.addWeighted(dmc, a, hsi, b, 0)
        plt.imshow(dst)
        plt.show()
        a += 0.2

for i, img_name in enumerate(file_list): 
    # if i < 450:
        # continue
    input("Press Enter to continue...")
    org_idx = os.path.splitext(os.path.basename(img_name))[0] 
    tar_idx = int(org_idx)
    img = cv2.imread(img_name)
    hsi = sio.loadmat(img_name.replace('jpg', 'mat'))['data']
    for k in range(999):
        if k>0:
            input_x=  input("offset (ex. 100 100), if satisfied enter y:")
            if input_x== 'y':
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
    plt.show()
    img[(cv2.resize(hsi_target, (1600,1600))==0)[:,:,0],:]=0
    plt.imshow(np.concatenate((img, hsi_view)))
    new_img_name = target_path + str(tar_idx) + '.jpg'
    new_hsi_name = target_path + str(tar_idx) + '.mat'
    new_hsi_jpg_name = target_path + 'low_res' +str(tar_idx) + '.jpg'
    cv2.imwrite(new_img_name, img)
    sio.savemat(new_hsi_name, {'data':hsi_target})
    cv2.imwrite(new_hsi_jpg_name, HSItensor2imgs(hsi_target))
    # for x, y in product(range(x_grid), range(y_grid)):
    #     new_img_name = target_path + str(tar_idx + i) + '.jpg'
    #     new_hsi_name = target_path + str(tar_idx + i) + '.mat'
    #     new_hsi_jpg_name = target_path + 'low_res' +str(tar_idx + i) + '.jpg'
    #     cv2.imwrite(new_img_name, img[jpg_grd[x]: jpg_grd[x] + jpg_size[0] , jpg_grd[y]: jpg_grd[y]+jpg_size[1], :])
    #     sio.savemat(new_hsi_name, {'data':hsi_target[hsi_grd[x]: hsi_grd[x] + hsi_size[0] , hsi_grd[y]: hsi_grd[y]+hsi_size[1], :]})
    #     cv2.imwrite(new_hsi_jpg_name, HSItensor2imgs(hsi_target[hsi_grd[x]: hsi_grd[x] + hsi_size[0] , hsi_grd[y]: hsi_grd[y]+hsi_size[1], :]))
    #     i+=1