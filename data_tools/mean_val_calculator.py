import scipy.io as sio
import argparse
import os
import glob
import numpy as np
import sys
from tqdm import tqdm
parser = argparse.ArgumentParser(description = 'Options')
parser.add_argument('--dir',
                   help='directory for mean and var calculation')
args = parser.parse_args()
data_dir = args.dir

file_names = glob.glob(data_dir + '*.mat')

mean = 0
var = 0
num_data = len(file_names)
for dat_idx in tqdm(range(num_data)):
    img = sio.loadmat(file_names[dat_idx])['data']
    mean = mean + np.mean(img[img[:,:,0]>0,:],axis=0)

mean = mean/num_data
print(mean)
print(mean.shape)
for dat_idx in tqdm(range(num_data)):
    img = sio.loadmat(file_names[dat_idx])['data']
    var = var + np.mean( (img[img[:,:,0]>0,:]-mean.reshape([1,-1]))**2, (0))

var = var/num_data
std = var**0.5
print(std)

sio.savemat('data/mean_std.mat',{"mean":mean, "std":std})
np.savetxt('data/mean.txt', mean.reshape([1,-1]), delimiter = ',' ,fmt='%1.4f')
np.savetxt('data/std.txt', std.reshape([1,-1]), delimiter = ',' ,fmt='%1.4f')