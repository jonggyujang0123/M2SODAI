#!/usr/bin/env python3
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

hsi_mat_name = '../data/test/230.mat'
jpg_name = '../data/test/230.jpg'

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
pixels_start_x_hsi = [48, 95, 64, 98, 72, 198]
pixels_end_x_hsi = [50, 99, 66, 100, 74, 200]
pixels_start_y_hsi = [88, 100, 151, 152, 72, 115]
pixels_end_y_hsi = [90, 101, 153, 154, 74, 117]

pixels_start_x_jpg = [357, 688, 617, 691, 500, 1390]
pixels_end_x_jpg = [365, 698, 625, 698, 515, 1400]
pixels_start_y_jpg = [640, 712, 1035, 1066, 500, 800]
pixels_end_y_jpg = [647, 721, 1043, 1074, 515, 815]

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
        pixels_start_y_jpg[i]-1:pixels_end_y_jpg[i]+1,
        pixels_start_x_jpg[i]-1:pixels_end_x_jpg[i]+1,
        :
        ])
    plt.show()
    cv2.imwrite(f'jpg_object_{i}.png', jpg[
        pixels_start_y_jpg[i]-1:pixels_end_y_jpg[i]+1,
        pixels_start_x_jpg[i]-1:pixels_end_x_jpg[i]+1,
        2::-1
        ])
# Plot reflectance response
## HSI
import seaborn as sns
import pandas as pd 
sns.set_theme(style="dark")
flights = sns.load_dataset("flights")
obj = list()
d_wl = list()
ref = list()
for i in range(len(pixels_start_x_hsi)):
    response = hsi[
        pixels_start_y_hsi[i]:pixels_end_y_hsi[i],
        pixels_start_x_hsi[i]:pixels_end_x_hsi[i],
        :
        ].copy().reshape([-1, 127])
    for pix_ind in range(response.shape[0]):
        for rad_ind in range(response.shape[1]):
            ref.append(response[pix_ind,rad_ind])
            d_wl.append(400 + rad_ind * (1000-400)/127)
            if i<2:
                obj.append(f'floating matter')
            elif i<4:
                obj.append(f'sea surface effect')
            else:
                obj.append(f'clean sea surface')
        
d = dict()
d['obj'] = obj
d['d_waveleng'] = d_wl
d['reflectance'] = ref
df = pd.DataFrame(data=d)

    
# Plot each year's time series in its own facet
g = sns.relplot(
    data=df,
    x="d_waveleng", y="reflectance", col="obj", hue="obj",
    kind="line", palette="crest", linewidth=3, zorder=5,
    col_wrap=3, height=4, aspect=1.5, legend=False
)

# Iterate over each subplot to customize further
for year, ax in g.axes_dict.items():

    # Add the title as an annotation within the plot
    ax.text(.1, .85, year, fontsize=22, transform=ax.transAxes, fontweight="bold")

    # Plot every year's time series in the background
    sns.lineplot(
        data=df, x="d_waveleng", y="reflectance", units="obj", zorder = 0,
        estimator=None, color=".7", alpha = 0.2, linewidth=1, ax=ax,
    )

# Reduce the frequency of the x axis ticks

# ax.set_xticks([200,400,600,800,1000])

# Tweak the supporting aspects of the plot
g.set_titles("")
g.set_axis_labels("Wavelength (nm)", "Reflectance (%/100)", fontsize=18)
g.tight_layout()
plt.savefig("ref.svg")
plt.show()

# Plot hsi cube
hsi = hsi-np.min(hsi,(0,1))
data = hsi/np.max(hsi,(0,1))
    
fig = plt.figure()
ax = fig.gca(projection='3d')

x = np.linspace(0, 1, 224)
X, Y = np.meshgrid(x, x)
Z = np.zeros_like(X)

levels = np.linspace(-1, 1, 40)

num_channel = 127
for i in tqdm(range(num_channel)):
    if i == num_channel-1:
        ax.plot_surface(X, Y, 0.1*i+Z, rstride=2, cstride=2,
                facecolors=jpg_hsi/255)
    else:
        ax.plot_surface(X, Y, 0.1*i+Z, rstride=4, cstride=4,
                facecolors=cm.coolwarm(data[:,:,i]))
        # ax.contourf(X, Y, 0.1*i+data[:,:,i], zdir='z', levels=0.1*i + .1*levels)
# ax.contourf(X, Y, 3+data[:,:,20], zdir='z', levels=3+.1*levels)
# ax.contourf(X, Y, 7+data[:,:,90], zdir='z', levels=7+.1*levels)

# ax.legend()
ax.set_xlim3d(0, 1)
ax.set_ylim3d(0, 1)
ax.set_zlim3d(0, 13)
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
ax.set_yticklabels([])
ax.set_xticklabels([])
ax.set_zticklabels([])

plt.show()

fig.savefig("cube.png")



