import numpy as np
import math
import pandas as pd
import glob
import matplotlib.pyplot as plt
from random import choice
import cv2
from tensorflow.python.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, GlobalAveragePooling2D
from tensorflow.python.keras.models import Sequential, load_model, Model
from tensorflow.python.keras.applications.xception import Xception
#from model import *
#from LSTM import *
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
from skimage.io import imread

test_file   = 'F:\\Glau\\Refuge2-Validation\\Refuge2-Validation\\'
    
def read_data(data_file):
    filename = glob.glob(data_file + '\\*.jpg')
    im_name = []
    for i in filename:
        n = i.split('\\')[-1].split('.')[0]
        im_name.append(n)
    data_split_out = []
    aa = []
    for file in im_name:
        data = imread(data_file + '\\' + file + '.jpg')
        data = cv2.resize(data,(512,512))
        data  = np.reshape(data,(1,)+data.shape)
        aa.append(data)            
    for k in range(0,len(aa)):
        ss = np.reshape(aa[k],(1,)+aa[k].shape).T
        data_split_out.append(np.reshape(ss,(1,)+ss.shape))
    return aa, im_name
def nor(list_,mode = 'nor'):
    out = []
    if mode == 'nor':
        for l in list_:
            l = l/255.
#            l = (l - l.min())/(l.max() - l.min())
            out.append(l)
    if mode == 'std':
        for l in list_:
            l = (l - l.mean())/l.std()
            out.append(l)
    return out
def pre_argmax(predict):
    ans = []
    for i in range(0,len(predict)):
        yy = np.argmax(predict[i])
        ans.append(yy)
    return ans

def pred(X):
    predict = []
    for i in range(0,len(X)):
        y_predict = model.predict(X[i])
        predict.append(y_predict)
    return predict

gx,nam1 = read_data(test_file)
#bx,nam2 = read_data(test_file)
#gx.extend(bx)
#nam1.extend(nam2)
X = nor(gx,'nor')

nb_class=1
base_model = Xception(weights='imagenet', include_top=False)
x = base_model.output
x = GlobalAveragePooling2D(data_format='channels_last')(x)
x = Dropout(0.5)(x)
predictions = Dense(nb_class, activation='sigmoid', name='classifier')(x)
model = Model(inputs=base_model.input, outputs=predictions)

model.load_weights('F:\\Glau\\saved_models\\Xceptionsig(crop).h5')
predict = []
predict = pred(X)
#ans = pre_argmax(predict)
result = []
for i in range(0,len(predict)):
    s = predict[i]
    ss = s[0]
    result.append([nam1[i]+'.jpg',ss[0]])
pds = pd.DataFrame(result)
pds.columns = ['FileName', 'Glaucoma Risk']
pds.to_csv('Classification_results(xception).csv', index=False)
    

