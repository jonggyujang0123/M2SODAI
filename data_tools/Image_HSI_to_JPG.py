#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  8 16:22:29 2022

@author: jgjang
"""

import numpy as np

from PIL import Image, ImageEnhance

def enhancer(img):
    PIL_img = Image.fromarray(img)
    out = ImageEnhance.Brightness(PIL_img).enhance(4)
    out = ImageEnhance.Contrast(out).enhance(0.7)
    out = ImageEnhance.Color(out).enhance(0.6)
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