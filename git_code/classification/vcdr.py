from skimage.io import imread,imshow,imsave
import numpy as np
import pandas as pd
import glob
import cv2
import os

im_files   = 'F:\\Glau\\Refuge2-Validation\\Refuge2-Validation'
disk_files = 'G:\\disc_ensemble' 
cup_files  = 'G:\\return_CLAHE_green_laplacian_ensemble' 

ims   = sorted(glob.glob(im_files+ '\\*.jpg'))
result = []
for im in ims:
    name = im.split('\\')[-1].split('.jpg')[0]
    image= imread(im)
    od_b = (255-imread(disk_files + '\\' + name + '.png'))/255
    oc_b = (255-imread(cup_files  + '\\' + name + '.png')[:,:,0])/255
    Aod  = np.where(od_b==1)
    
    rc   = np.sqrt(np.sum(oc_b)/np.pi)
    rd   = np.sqrt(np.sum(od_b)/np.pi)
#    if os.path.isfile('D:\\glaucoma\\img\\crop\\train\\glaucoma\\' + name + '.jpg'):
#        result.append([name,rc,rd,rc/rd,1])
#    else:
#        result.append([name,rc,rd,rc/rd,0])
    result.append([name+'.jpg',rc/rd])
pds = pd.DataFrame(result)
pds.columns = ['FileName', 'Glaucoma Risk']
pds.to_csv('Classification_results(vcdr).csv', index=False)
#np.save('.\\cup_disk_rate_val',result)