import glob
import numpy as np
from skimage.io import imread,imsave
from skimage.transform import resize
import cv2



im_path=r'/media/yared/HV620S/Blood(REFUGE Data)/train/total'
im_filename = glob.glob(im_path + '/*.jpg')
im_name = []
for i in im_filename:
    n = i.split('/')[-1].split('.')[0]
    im_name.append(n)

for im_name in im_name:
    pic = imread(im_path+'/'+im_name+'.jpg')
    # # pic =cv2.cvtColor(pic,cv2.COLOR_BGR2GRAY)
    # m,n,o = pic.shape
    # image_a = np.zeros([m,n,o])
    # level_array = np.zeros((256))
    # Sr = np.zeros((256))
    # for i in range (m):
    #     for j in range (n):
    #         for k in range(o):
    #             a = pic[i,j,k]
    #         level_array[a] = level_array[a]+1
    # Sr[0] = level_array[0]
    # for k in range (1,256):
    #     Sr[k] = Sr[k-1]+level_array[k]
    # Sr = Sr*255/(m*n)   
    # Sr_N = np.round(Sr)
    # for i in range (m):
    #     for j in range(n):
    #         for k in range(o):
    #             pixel = pic[i,j,k]
    #             image_a[i,j,k]  = Sr_N[pixel]
    # img=image_a.astype(np.uint8)
    # img2 = cv2.cvtColor(image_a,cv2.COLOR_GRAY2BGR)
    lab = cv2.cvtColor(pic, cv2.COLOR_BGR2LAB)

    lab_planes = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=10,tileGridSize=(8,8))

    lab_planes[0] = clahe.apply(lab_planes[0])

    lab = cv2.merge(lab_planes)

    img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
    cv2.imwrite(r'/media/yared/HV620S/Blood(REFUGE Data)/train/' + im_name.split('/')[-1].split('.')[0] + '.jpg',img)    

