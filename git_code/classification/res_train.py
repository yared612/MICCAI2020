import tensorflow as keras
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.models import Sequential, load_model, Model
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
from tensorflow.python.keras.models import Sequential, load_model, Model
from tensorflow.python.keras.applications.xception import Xception
from tensorflow.python.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, GlobalAveragePooling2D
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cv2
import glob
from skimage.io import imread


train_file = 'F:\\Glau\\img\\crop\\aug\\train'
val_file   = 'F:\\Glau\\img\\crop\\aug\\val'
class_ = glob.glob(train_file + '\\*')
class_name = []
for name in class_:
    class_name.append(name.split('\\')[-1])

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
#        data = data[:,:,1]
        data = cv2.resize(data,(299,299))/255.
#        dd = np.zeros((512,512,3))
#        dd[:,:,0] = data
#        dd[:,:,1] = data
#        dd[:,:,2] = data
        aa.append(data)            
    for k in range(0,len(aa)):
        ss = np.reshape(aa[k],(1,)+aa[k].shape)
        data_split_out.append(np.reshape(ss,ss.shape + (1,)))
    return aa
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
bX = read_data(train_file + '\\' + class_name[0] )
gX = read_data(train_file + '\\' + class_name[1] )
val_bX = read_data(val_file + '\\' + class_name[0] )
val_gX = read_data(val_file + '\\' + class_name[1] )
good_y = np.zeros([len(gX)])
bad_y  = np.ones([len(bX)])
train_y_z = np.concatenate([good_y,bad_y])
#train_y_o = np.stack([1-train_y_z,train_y_z],axis=-1)
val_good_y = np.zeros([len(val_gX)])
val_bad_y  = np.ones([len(val_bX)])
val_y_z = np.concatenate([val_good_y,val_bad_y])
#val_y_o = np.stack([1-val_y_z,val_y_z],axis=-1)
gX.extend(bX), val_gX.extend(val_bX)
com = []
for i in range (0,len(gX)):
    com.append([gX[i],train_y_z[i]])
com = shuffle(com)
X = []
y = []
for j in range(0,len(com)):
    ss = com[j]
    X.append(ss[0])
    y.append(ss[1])
X, y, val_X, val_y = X , y , val_gX , val_y_z


nb_class=1
img_rows, img_cols, img_channel = 299, 299, 3
base_model = Xception(weights='imagenet', include_top=False,
                      input_shape=(img_rows, img_cols, img_channel))
x = base_model.output
x = GlobalAveragePooling2D(data_format='channels_last')(x)
x = Dropout(0.5)(x)
predictions = Dense(nb_class, activation='sigmoid', name='classifier')(x)
model = Model(inputs=base_model.input, outputs=predictions)

model.compile(optimizer=Adam(lr=1e-5), loss='binary_crossentropy', metrics=['accuracy'])
epochs = 300
saved_dir = './saved_models'
model_name = 'Xceptionsig(crop1).h5'
model_path = '/'.join((saved_dir, model_name))
checkpoint = ModelCheckpoint(model_path, monitor='val_loss', verbose=1,save_best_only = True)
EarlyStopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=50, verbose=1)
csv_logger = CSVLogger('Xcptionsig(crop1).log')
weights = {0: 0.3,
           1: 1}
datagen = ImageDataGenerator(
        rotation_range=10,
#        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest')
batch_size = 4
model_history = model.fit_generator(datagen.flow(np.array(X), np.array(y), batch_size = batch_size),
                                    epochs = epochs,
                                    class_weight = weights,
                                    validation_data = (np.array(val_X), np.array(val_y)),
                                    callbacks = [checkpoint, EarlyStopping])

training_acc = model_history.history['acc']
val_acc = model_history.history['val_acc']
plt.plot(training_acc, label="training_acc")
plt.plot(val_acc, label="validation_acc")
plt.xlabel("Epochs")
plt.ylabel("Acc")
plt.title("Learning Curve")
plt.legend(loc='best')
plt.show()

training_loss = model_history.history['loss']
val_loss = model_history.history['val_loss']
plt.plot(training_loss, label="training_loss")
plt.plot(val_loss, label="validation_loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Learning Curve")
plt.legend(loc='best')
plt.show()