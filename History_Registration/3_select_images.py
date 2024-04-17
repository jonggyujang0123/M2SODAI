import os
import glob
import argparse
import cv2
import numpy as np
from matplotlib import pyplot as plt
import shutil

parser = argparse.ArgumentParser(description = 'Options')
parser.add_argument('--dir', help = 'directory of dataset')
args = parser.parse_args()

path = args.dir

dir_name = '/data_save_v2/'
tar_name =  '/data_save_selected/'

img_path = path + dir_name
target_path = path + tar_name

if not os.path.exists(target_path):
    os.makedirs(target_path)


file_list = sorted(glob.glob(img_path + '*[!_hsi].jpg'))

for i, img_name in enumerate(file_list):
    if i < 1101:
        continue
    # define file names 
    img_name = img_name
    mat_name = img_name.replace('.jpg','.mat')
    hsi_name = img_name.replace('.jpg', '_hsi.jpg')
    # load jpg image
    img = cv2.imread(img_name)
    # load hsi image
    hsi = cv2.imread(hsi_name)
    # concat
    img_hsi = np.concatenate([
        img[:,:,2::-1], 
        cv2.resize(hsi, img.shape[1::-1])],axis=1)
    plt.imshow(img_hsi)
    plt.show(block=False)
    while True:
        input_x=  input(f"{i}/{len(file_list)}-th data, if selected, enter y, otherwise n:")
        if input_x in ['y', 'n']:
            break
    plt.close()
    if input_x == 'y':
        # Move files 
        shutil.copy2(img_name, img_name.replace(dir_name, tar_name))
        shutil.copy2(mat_name, mat_name.replace(dir_name, tar_name))
        shutil.copy2(hsi_name, hsi_name.replace(dir_name, tar_name))
