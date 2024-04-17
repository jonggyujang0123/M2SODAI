import glob
import cv2 
import argparse
import os
import spectral
from pathlib import Path
import numpy as np
import spectral.io.envi as envi
from spectral import *
import numpy as np
import cv2
import scipy.io as sio
parser = argparse.ArgumentParser(description = 'Options')
parser.add_argument('--dir', help = 'directory of HSI folder')
args = parser.parse_args()

path = args.dir

max_col = 224 # Cropping grid size(column)
max_row = 224 # Cropping grid size(row)


from PIL import Image, ImageEnhance

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


if __name__ == "__main__":
    hdr_name_list = sorted(glob.glob(path+ 'HSIs/*.hdr'))
    bsq_name_list = sorted(glob.glob(path+ 'HSIs/*.bsq'))
    
    if not os.path.exists(path+'HSI_DATA'):
        os.mkdir(path+'HSI_DATA')
    
    for file_ind in range(len(hdr_name_list)):
        #if file_ind < 8:
        #    continue
        print(hdr_name_list[file_ind])
        hdr_name = hdr_name_list[file_ind]
        bsq_name = bsq_name_list[file_ind]
        path_str = Path(os.path.splitext(hdr_name)[0]).parts
        img = envi.open(hdr_name,bsq_name)
        arr = img.load()
        img_ndvi_R = ndvi(arr + 1e-5,50,86)
        #if not os.path.exists(path+'HSI_DATA'):
        #    os.mkdir(path+'HSI_DATA')
        hsi = HSItensor2imgs(arr)
        cv2.imwrite(f'{path}/HSIs/{path_str[-1]}.jpg',hsi)
        #save_rgb(path+'HSI_DATA/'+path_str[-1] +'.jpg',arr,[58,37,17])
        #save_rgb(path+'HSI_DATA/'+path_str[-1] +'_ndvi.jpg',img_ndvi_R,stretch=0.001)
        #img_ndvi_R = cv2.imread(path+'HSI_DATA/' + path_str[-1] +'_ndvi.jpg')
        #hsi = cv2.imread(path+ path_str[-1] +'.jpg')
        
        # col_num = int(np.floor(img.shape[0]/max_col))
        # row_num = int(np.floor(img.shape[1]/max_row))
        # for col_ind in range(col_num):
        #     for row_ind in range(row_num):
        #         c_start = col_ind * max_col; c_end = (col_ind+1)*max_col
        #         r_start = row_ind * max_row; r_end = (row_ind+1)*max_row
        #         if np.sum(np.sum(hsi[c_start:c_end, r_start:r_end,:],axis=2)==0) > 0.25* max_row * max_col:
        #             continue
        #         path_f_save =f'{path}/HSI_DATA/{path_str[-1]}_{col_ind}_{row_ind}'
        #         cv2.imwrite(f'{path_f_save}_hsi.jpg',
        #                 hsi[c_start:c_end, r_start:r_end,:],
        #                 )
        #         save_rgb(f'{path_f_save}_ndvi_R.jpg',
        #                 img_ndvi_R[c_start:c_end, r_start:r_end], 
        #                 )
        #         sio.savemat(f'{path_f_save}_hsi.mat',
        #                 {'data':arr[c_start:c_end, r_start:r_end,:]})


