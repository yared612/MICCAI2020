import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import cv2
import glob
import pandas as pd
from skimage.io import imread,imsave,imshow
from skimage.transform import resize
from time import time


im_path = '/home/user/Documents/median_never/ori_img/stage1_test'
#save_path = '/home/john/Desktop/REFUGE/Refuge2-Validation/LDF'
im_filename = glob.glob(im_path + '/*.bmp')
im_name = []
for i in im_filename:
    n = i.split('/')[-1].split('.')[0]
    im_name.append(n)
im_name.sort()    
def read_data(im_name):
    for i in im_name:
        I = imread(im_path + '/' + i + '.bmp')
        kernel_size = (51, 51);
        sigma = 1;
        G = cv2.GaussianBlur(I, kernel_size, sigma);
        A = cv2.blur(I, (51, 51))
        pic = (G - A)
    return pic,I
#        imsave(save_path + '/' + i + '.jpg', pic)
x,img = read_data(im_name[1:2])
imshow(x)
        