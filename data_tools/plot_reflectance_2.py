#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 16:35:32 2022

@author: jgjang
"""

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

## HSI
import seaborn as sns
import pandas as pd 
import os

if not os.path.exists('./samples'):
    os.mkdir('./samples')

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

global data_collection, data_ship, data_float, data_sur_eff, data_clean

data_ship = list()
data_float = list()
data_sur_eff = list()
data_clean = list()
data_collection = (data_ship,
                   data_float,
                   data_sur_eff,
                   data_clean)

def data_crop(data_ind, print_=False):
    if data_ind == 230:
        pixels_start_x_hsi = [48, 95, 64, 98, 72, 198]
        pixels_end_x_hsi = [50, 99, 66, 100, 82, 200] 
        pixels_start_y_hsi = [88, 100, 151, 152, 72, 115]
        pixels_end_y_hsi = [90, 101, 153, 154, 82, 117]

        pixels_start_x_jpg = [356, 687, 616, 690, 549, 1389]
        pixels_end_x_jpg = [366, 699, 626, 699, 609, 1401]
        pixels_start_y_jpg = [639, 711, 1034, 1065, 499, 799]
        pixels_end_y_jpg = [648, 722, 1044, 1075, 559, 816]
        
        sep = [0, 2, 4, 6] # ship, float, sur_eff, data_clean
    if data_ind ==0:
        pixels_start_x_hsi = [2]
        pixels_end_x_hsi = [23] 
        pixels_start_y_hsi = [217]
        pixels_end_y_hsi = [223]

        pixels_start_x_jpg = [0]
        pixels_end_x_jpg = [230]
        pixels_start_y_jpg = [1420]
        pixels_end_y_jpg = [1600]
        sep = [1, 1, 1, 1]
    if data_ind ==260:
        pixels_start_x_hsi = [32, 77, 147, 130, 35]
        pixels_end_x_hsi = [36, 81, 150, 133, 39] 
        pixels_start_y_hsi = [13, 79, 167, 185, 20]
        pixels_end_y_hsi = [14, 81, 169, 187, 24]

        pixels_start_x_jpg = [240, 688, 1379, 1255, 260]
        pixels_end_x_jpg = [259, 705, 1402, 1272, 290]
        pixels_start_y_jpg = [83, 677, 1220, 1432, 132]
        pixels_end_y_jpg = [100, 693, 1240, 1452, 170]
        sep = [0, 1, 4, 5]
    if data_ind == 344:
        pixels_start_x_hsi = [175, 132, 201, 155, 90, 126]
        pixels_end_x_hsi = [193, 147, 204, 158, 93, 130] 
        pixels_start_y_hsi = [48, 76, 85, 125, 46, 19]
        pixels_end_y_hsi = [55, 86, 87, 127, 49, 23]

        pixels_start_x_jpg = [1225, 890, 1112, 1426, 248, 816]
        pixels_end_x_jpg = [1410, 1120, 1128, 1438, 258, 828]
        pixels_start_y_jpg = [290, 500, 888, 596, 1198, 334]
        pixels_end_y_jpg = [395, 630, 902, 609, 1205, 341]
        sep = [2, 4, 6, 6]
    if data_ind == 197:
        pixels_start_x_hsi = [42, 26,  9]
        pixels_end_x_hsi =   [47, 29, 12] 
        pixels_start_y_hsi = [100, 66, 98]
        pixels_end_y_hsi =   [123, 69, 100]

        pixels_start_x_jpg = [280, 189, 77]
        pixels_end_x_jpg =   [370, 205, 87]
        pixels_start_y_jpg = [695, 468, 694]
        pixels_end_y_jpg =   [885, 481, 711]
        sep = [1, 3, 3, 3]
    if data_ind == 247:
        pixels_start_x_hsi = [130, 19, 70, 82, 28, 130, 150]
        pixels_end_x_hsi =   [143, 30, 78, 89, 36, 140, 160] 
        pixels_start_y_hsi = [110, 108,72, 45, 52, 20, 30]
        pixels_end_y_hsi =   [140, 129,98, 58, 66, 30, 40]

        pixels_start_x_jpg = [920, 480, 150, 70, 550, 1100, 1250]
        pixels_end_x_jpg =   [1030, 560, 280, 280, 640, 1160, 1310]
        pixels_start_y_jpg = [760, 500, 325, 740, 295, 160, 230]
        pixels_end_y_jpg =   [1020, 700, 500, 967, 400, 220, 290]
        sep = [5, 5, 5, 7]
    if data_ind == 144:
        pixels_start_x_hsi = [155, 195, 80]
        pixels_end_x_hsi =   [170, 198, 90]
        pixels_start_y_hsi = [112, 112, 40]
        pixels_end_y_hsi =   [119, 114, 50]

        pixels_start_x_jpg = [1080, 1395, 1200]
        pixels_end_x_jpg =   [1240, 1408, 1260]
        pixels_start_y_jpg = [780, 798, 310]
        pixels_end_y_jpg =   [885, 812, 370]
        sep = [1, 2, 2, 3]
    if data_ind == 263:
        pixels_start_x_hsi = [150, 150, 130, 83, 78, 161, 190]
        pixels_end_x_hsi =   [153, 153, 133, 86, 81, 164, 200]
        pixels_start_y_hsi = [15, 78, 154, 191, 195, 80, 183]
        pixels_end_y_hsi =   [18, 81, 157, 193, 197, 82, 193]

        pixels_start_x_jpg = [1060, 1072, 944, 1532, 1270, 978, 1290]
        pixels_end_x_jpg =   [1074, 1084, 954, 1548, 1286, 988, 1350]
        pixels_start_y_jpg = [105, 558, 1110, 1293, 1308, 1375, 1240]
        pixels_end_y_jpg =   [122, 572, 1128, 1308, 1316, 1384, 1300]
        sep = [0, 3, 6, 7]
    hsi_mat_name = f'../data/test/{data_ind}.mat'
    jpg_name = f'../data/test/{data_ind}.jpg'
    hsi = sio.loadmat(hsi_mat_name)['data'] # (224, 224, 127)
    jpg = cv2.imread(jpg_name) # (1600, 1600, 3)
    jpg_hsi = HSItensor2imgs(hsi)[:,:,2::-1]
    for i in range(len(pixels_end_x_hsi)):
        list_tmp = (hsi[pixels_start_y_hsi[i]:pixels_end_y_hsi[i],
                    pixels_start_x_hsi[i]:pixels_end_x_hsi[i],
                    :],
                jpg[pixels_start_y_jpg[i]:pixels_end_y_jpg[i],
                    pixels_start_x_jpg[i]:pixels_end_x_jpg[i],
                    :]
                )
        if i<sep[0]:
            data_ship.append(list_tmp)
        elif i<sep[1]:
            data_float.append(list_tmp)
        elif i<sep[2]:
            data_sur_eff.append(list_tmp)
        else:
            data_clean.append(list_tmp)
    if print_:
        ## HSI
        for i in range(len(pixels_start_x_hsi)):
            plt.imshow(jpg_hsi[
                pixels_start_y_hsi[i]:pixels_end_y_hsi[i],
                pixels_start_x_hsi[i]:pixels_end_x_hsi[i],
                :
                ])
            plt.show()
            plt.imshow(jpg[
                pixels_start_y_jpg[i]:pixels_end_y_jpg[i],
                pixels_start_x_jpg[i]:pixels_end_x_jpg[i],
                :
                ])
            plt.show()

################################################################################### data : 230

indices = [230, 0, 260, 344, 197, 247, 144, 263]
for i in indices:
    data_crop(i, print_=False)

## Plot reflectance response

obj = list()
d_wl = list()
ref = list()


for type_ind in range(len(data_collection)):
    if type_ind ==0:
        obj_ = 'ship'
    elif type_ind == 1:
        obj_ = 'floating matter'
    elif type_ind == 2:
        obj_ = 'sea surface effect'
    else:
        obj_ = 'clean sea surface'
    for i in range(len(data_collection[type_ind])):
        response = data_collection[type_ind][i][0].reshape([-1, 127])
        rgb_img = data_collection[type_ind][i][1]
        cv2.imwrite(f'./samples/jpg_object_{obj_}_{i}.png', rgb_img[:,:,2::-1])
        for pix_ind in range(response.shape[0]):
            for ref_ind in range(response.shape[1]):
                ref.append(response[pix_ind, ref_ind])
                d_wl.append(400 + ref_ind * (1000-400)/127)
                obj.append(obj_)

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
    col_wrap=4, height=3, aspect=1.5, legend=False
)

# Iterate over each subplot to customize further
for year, ax in g.axes_dict.items():

    # Add the title as an annotation within the plot
    ax.text(.1, .85, year, fontsize=20, transform=ax.transAxes, fontweight="bold")
    ax.set(ylim=(0, 0.22))
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
plt.savefig("ref_2.svg")
plt.show()


#############################################################################

#######################Similarity HSI#############################################


indices = ('ship',
           'floating matter',
           'sea surface effect',
           'clean sea surface')
A = pd.DataFrame(index =indices, columns=indices, dtype=float)


mean = np.zeros([4, 127])
for x_ind, x_axis in enumerate(indices):
    x_data = np.zeros([0,127])
    #load data
    for i in range(len(data_collection[x_ind])):
        x_data = np.concatenate([x_data, data_collection[x_ind][i][0].reshape([-1,127])],axis=0)
    mean[x_ind,:] = x_data.mean(axis=0)

mean = mean.mean(axis=0, keepdims=True)

for x_ind, x_axis in enumerate(indices):
    #load data
    x_data = np.zeros([0,127])
    for i in range(len(data_collection[x_ind])):
        x_data = np.concatenate([x_data, data_collection[x_ind][i][0].reshape([-1,127])],axis=0)
    for y_ind, y_axis in enumerate(indices):
        #load data
        y_data = np.zeros([0,127])
        for j in range(len(data_collection[y_ind])):
            y_data = np.concatenate([y_data, data_collection[y_ind][j][0].reshape([-1,127])],axis=0)
        # cos = np.matmul(x_data, np.transpose(y_data))
        # cos = cos / ( np.linalg.norm(x_data,axis=1,keepdims=True) * 
        #              np.linalg.norm(y_data.transpose(),axis=0,keepdims=True))
        # A[x_axis][y_axis]= cos.mean()
        
        cos = np.matmul(x_data.mean(axis=0,keepdims=True)-mean, np.transpose(y_data.mean(axis=0,keepdims=True)-mean))
        cos = cos / ( np.linalg.norm(x_data.mean(axis=0,keepdims=True)-mean,axis=1,keepdims=True) * 
                      np.linalg.norm((y_data.mean(axis=0,keepdims=True)-mean).transpose(),axis=0,keepdims=True))
        A[x_axis][y_axis]= cos.mean()


# 그림 사이즈 지정
fig, ax = plt.subplots( figsize=(4,4) )

# 삼각형 마스크를 만든다(위 쪽 삼각형에 True, 아래 삼각형에 False)
mask = np.zeros_like(A, dtype=np.bool)
mask[np.triu_indices_from(mask, k=1)] = True

# 히트맵을 그린다
heatmap = sns.heatmap(A, 
            cmap = 'RdYlBu_r', 
            annot = True,   # 실제 값을 표시한다
            mask=mask,      # 표시하지 않을 마스크 부분을 지정한다
            linewidths=.5,  # 경계면 실선으로 구분하기
            cbar_kws={"shrink": .5},# 컬러바 크기 절반으로 줄이기
            vmin = -1,vmax = 1   # 컬러바 범위 -1 ~ 1
           )  
loc, labels = plt.xticks()
heatmap.set_xticklabels(labels, rotation=45)
heatmap.set_yticklabels(labels, rotation=45) # reversed order for y
plt.savefig("heat_map_HSI.eps", bbox_inches = 'tight', format= 'eps', dpi=1000)

plt.show()


#############################################################################

#######################Similarity RGB#############################################

# !pip install -U sentence-transformers
from sentence_transformers import SentenceTransformer, util
from PIL import Image


indices = ('ship',
           'floating matter',
           'sea surface effect',
           'clean sea surface')
similarity = pd.DataFrame(data =0, index =indices, columns=indices, dtype=float)
visit = pd.DataFrame(data =0, index =indices, columns=indices, dtype=float)


model = SentenceTransformer('clip-ViT-B-32')

## Obtain listed images
RGB_imgs = list()
sep = np.cumsum(list(map(lambda x: len(x), data_collection)))
for x_ind, x_axis in enumerate(indices):
    for i in range(len(data_collection[x_ind])):
        RGB_imgs.append(Image.fromarray(data_collection[x_ind][i][1]))

encoded_img = model.encode([img for img in RGB_imgs], batch_size = 128, convert_to_tensor=True, show_progress_bar=True)
processed_images = util.paraphrase_mining_embeddings(encoded_img )

for score, image_id1, image_id2 in processed_images:
    name_1 = indices[(image_id1 - sep >=0).sum()]
    name_2 = indices[(image_id2 - sep >=0).sum()]
    print("\nScore: {:.3f}%".format(score * 100))
    print(name_1)
    print(name_2)
    similarity[name_1][name_2] += score
    visit[name_1][name_2] += 1

# 그림 사이즈 지정
fig, ax = plt.subplots( figsize=(4,4) )

# 삼각형 마스크를 만든다(위 쪽 삼각형에 True, 아래 삼각형에 False)
mask = np.zeros_like(A, dtype=np.bool)
mask[np.triu_indices_from(mask, k=1)] = True

# 히트맵을 그린다
heatmap = sns.heatmap(similarity/visit, 
            cmap = 'RdYlBu_r', 
            annot = True,   # 실제 값을 표시한다
            mask=mask,      # 표시하지 않을 마스크 부분을 지정한다
            linewidths=.5,  # 경계면 실선으로 구분하기
            cbar_kws={"shrink": .5},# 컬러바 크기 절반으로 줄이기
            vmin = 0,vmax = 1   # 컬러바 범위 -1 ~ 1
           )  
loc, labels = plt.xticks()
heatmap.set_xticklabels(labels, rotation=45)
heatmap.set_yticklabels(labels, rotation=45) # reversed order for y
plt.savefig("heat_map_RGB.eps", bbox_inches = 'tight', format= 'eps', dpi=1000)

plt.show()
