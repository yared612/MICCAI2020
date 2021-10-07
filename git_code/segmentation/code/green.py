# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 13:33:22 2020

@author: 20190524
"""
import glob
import cv2
from skimage.io import imread,imsave,imshow
import numpy as np 
import matplotlib.pyplot as plt
im_name=[]
im_path = (r'/media/yared/HV620S/Blood(REFUGE Data)/REFUGE2/Refuge2-Validation')
# oc_path = (r'D:\青光眼\return_green_laplacian')
im_filename = glob.glob(im_path + '/*.jpg')

for q in im_filename:
    n = q.split('/')[-1].split('.')[0]
    im_name.append(n)
for q in im_name :
    color =imread(im_path +'/' + q + '.jpg')
    r, g, b = cv2.split(color)
    ss = q.split('/')[-1].split('.')[0]
    
    imsave(r'/media/yared/HV620S/Blood(REFUGE Data)/REFUGE2/green-blood/'  +str(ss)+ '.jpg',g) 

