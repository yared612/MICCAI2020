from skimage.io import imread,imshow,imsave
import numpy as np
import pandas as pd
import glob
import cv2
import os

im_files   = 'F:\\Glau\\all_img(odc)'
disk_files = 'F:\\Glau\\OD' 
#cup_files  = 'C:\\Users\\tommy\\Desktop\\REFUGE_Fovea\\OC' 

ims   = sorted(glob.glob(im_files+ '\\*.jpg'))
result = []
for im in ims:
    name = im.split('\\')[-1].split('.jpg')[0]
    image= imread(im)
    od_b = (255-imread(disk_files + '\\' + name + '.bmp'))/255
#    oc_b = (255-imread(cup_files  + '\\' + name + '.bmp'))/255
#    rc   = np.sqrt(np.sum(oc_b)/np.pi)
    rd   = np.sqrt(np.sum(od_b)/np.pi)
    result.append(rd)

    
    