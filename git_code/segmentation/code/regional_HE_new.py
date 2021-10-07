import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "7"
import glob
import numpy as np
from skimage.io import imread,imsave,imshow
from skimage.transform import resize
import cv2

im_path =(r'/home/yared/文件/Blood(REFUGE Data)/train/total/')
gt_path = (r'/home/yared/文件/gt-total/') 
im_filename = glob.glob(im_path+'/*.jpg')
im_name = []
for i in im_filename:
    n = i.split('/')[-1].split('.')[0]
    im_name.append(n)
for im_name in im_name:
    img = cv2.imread(im_path+'/'+im_name+'.jpg')
    gt = cv2.imread(gt_path+'/'+im_name+'.bmp')
    m,n,o = img.shape
    image_a = np.zeros([m,n,o])
    level_array1 = np.zeros((256))
    level_array2 = np.zeros((256))
    Sr1 = np.zeros((256))
    Sr2 = np.zeros((256))
    for i in range (m):
        for j in range (n):
            for k in range(o):
                if gt[i,j,k] == 0:
                    a = img[i,j,k]
                    level_array1[a] = level_array1[a]+1
                else:
                    b = img[i,j,k]
                    level_array2[b] = level_array2[b]+1
                
    Sr1[0] = level_array1[0]
    Sr2[0] = level_array2[0]
    for k in range (1,256):
        Sr1[k] = Sr1[k-1]+level_array1[k]
    for k in range (1,256):
        Sr2[k] = Sr2[k-1]+level_array2[k]    
    Sr1 = Sr1*255/np.sum((255-gt)/255)   
    Sr1_N = np.round(Sr1)
    Sr2 = Sr2*128/np.sum(gt/255)
    Sr2_N = np.round(Sr2)
    for i in range (m):
        for j in range(n):
            for k in range(o):
                if gt[i,j,k] == 0:
                    pixel = img[i,j,k]
                    image_a[i,j,k]  = Sr1_N[pixel]
                else:
                    pixcel = img[i,j,k]
                    image_a[i,j,k] = Sr2_N[pixcel]
    img=image_a.astype(np.uint8)  
    cv2.imwrite(r'/home/yared/文件/Blood(REFUGE Data)/train/HE/' + im_name.split('/')[-1].split('.')[0] + '.jpg',img)
