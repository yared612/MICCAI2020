from skimage.io import imread, imshow ,imsave
import matplotlib.pyplot as plt
from skimage.transform import rotate 
import glob

train_files = 'F:\\Glau\\img\\crop\\train'
val_files = 'F:\\Glau\\img\\crop\\val' 
class_ = glob.glob(train_files + '\\*')
class_name = []
for name in class_:
    class_name.append(name.split('\\')[-1])

tb_ims = sorted(glob.glob(train_files + '\\' + class_name[0] + '\\*.jpg'))
vb_ims = sorted(glob.glob(val_files+ '\\' + class_name[0] + '\\*.jpg'))
tg_ims = sorted(glob.glob(train_files + '\\' + class_name[1] + '\\*.jpg'))
vg_ims = sorted(glob.glob(val_files+ '\\' + class_name[1] + '\\*.jpg'))
tbn = []
vbn = []
tgn = []
vgn = []
for i in tb_ims:
    n = i.split('\\')[-1].split('.')[0]
    tbn.append(n)
for i in vb_ims:
    n1 = i.split('\\')[-1].split('.')[0]
    vbn.append(n1)
for i in tg_ims:
    n2 = i.split('\\')[-1].split('.')[0]
    tgn.append(n2)
for i in vg_ims:
    n3 = i.split('\\')[-1].split('.')[0]
    vgn.append(n3)

for i in range(0,11):
    tb = imread(train_files + '\\' + class_name[0] + '\\' + )
    
    imsave('F:\\Glau\\img\\crop\\aug\\train\\' + class_name[0] + '\\' + ,img)

