import os
import numpy as np
import cv2
import scipy.io as sio
import glob
from itertools import product
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser(description='HSI_dir')
parser.add_argument('--dir', type=str, default='data/train_pca', help='hsi_dir')
args = parser.parse_args()
img_path = args.dir

file_list = sorted(glob.glob(img_path + '/*.mat'))

img_path_target = args.dir.replace('pca', '_pca_vis')


if not os.path.exists(img_path_target):
    os.makedirs(img_path_target)


for file in tqdm(file_list):
    img = sio.loadmat(file)['data']
    #  clip_val = 40.0
    img /= 1.0
    #  img = np.clip(img, -clip_val, clip_val) / clip_val
    #  img = (img + 1) / 2
    #  img = (img + 1) / 2
    #  _max = np.max(np.max(img, axis=0, keepdims=True), axis=1, keepdims=True)
    #  _min = np.min(np.min(img, axis=0, keepdims=True), axis=1, keepdims=True)
    #  img = (img - _min) / (_max - _min)
    #  _mean = np.mean(np.mean(img, axis=0, keepdims=True), axis=1, keepdims=True)
    #  _std = np.std(img, axis=(0, 1), keepdims=True)
    #  img = (img - _mean)
    #  img = np.abs(img)


    img = np.tanh(img) 
    img = (img + 1) / 2
    #  img = (img + 1) / 2

    for i in range(30): #img.shape[2]):
        img[:, :, i] = img[:, :, i] * 255
        cv2.imwrite(
                img_path_target + '/' + os.path.basename(file).replace('.mat', f'_hsi_{i}_f.png'),
                np.uint8(img[:, :, i]))
    print(file)
    #  break

