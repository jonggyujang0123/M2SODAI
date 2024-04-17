import glob
import tifffile as tiff
import cv2
import argparse
import os
from pathlib import Path
import numpy as np
from PIL import Image

parser = argparse.ArgumentParser(description = 'Options')
parser.add_argument('--dir', help = 'directory of Tiff folder')
args = parser.parse_args()

path = args.dir

def main():
    file_list= glob.glob(path + '/DMC/*.tif',recursive=True)
    print(file_list)
#    if not os.path.exists(path+'JPG'):
#        os.mkdir(path+'JPG')



    for file_ind in range(len(file_list)):
        print(file_list[file_ind])
        im = tiff.imread(file_list[file_ind])
        #im = im[:,:,2::-1]
        im = Image.fromarray(im)
        #im = cv2.rotate(im, cv2.ROTATE_90_CLOCKWISE)
        #col_num = int(np.floor(np.maximum(0, im.shape[0] - max_col_h -1)/max_col_h)+1)
        #row_num = int(np.floor(np.maximum(0, im.shape[1] - max_row_h -1)/max_row_h)+1)
        width, height = im.size
        print(f'width : {width*0.1/1000}, height: {height*0.1/1000}')
        im = im.resize((width//7, height//7), Image.ANTIALIAS)
        #path_str = Path(os.path.splitext(file_list[file_ind])[0]).parts
        path_str = os.path.splitext(file_list[file_ind])[0]
#        im.save(f'{path}JPG/{path_str[-2]}_{path_str[-1]}.jpg')
        im.save(f'{path_str}.jpg')
#        path_str = Path(os.path.splitext(file_list[file_ind])[0]).parts
#        #print(path_str[-1], path_str[-2])
#        for col_ind in range(col_num):
#            for row_ind in range(row_num):
#                cv2.imwrite(path+'JPG/'+ path_str[-2] + '_' + path_str[-1] + '_' + str(col_ind) + '_'+ str(row_ind)+ '.jpg' , im[col_ind * max_col_h: (col_ind+2)* max_col_h ,row_ind * max_row_h : (row_ind+2)*max_row_h ,:])

if __name__ == "__main__":
    main()
    
