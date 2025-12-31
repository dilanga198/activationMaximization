# -*- coding: utf-8 -*-


import keras
import math
import numpy as np
import matplotlib.pyplot as plt
import keras.backend as K
import tensorflow as tf

from keras.datasets import mnist
from keras.models import Sequential,Model
from keras.layers import Dense,Conv2D,Flatten,MaxPooling2D,GlobalAveragePooling2D
from keras.utils import plot_model
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import Model
from matplotlib import pyplot
from numpy import expand_dims
from skimage import io
from keras.models import model_from_json
from keras.models import load_model

(X_train,Y_train),(X_test,Y_test)  = mnist.load_data()

#X_train = np.load('X_train.npy', allow_pickle=True)
#Y_train = np.load('Y_train.npy', allow_pickle=True)

X_test = X_test.reshape((X_test.shape[0],X_test.shape[1],X_test.shape[2],1))
X_test  = X_test/255
X_test  = X_test.astype('float')

X_test = X_test[:3000,:,:,:]


json_file = open('model2.json', 'r')

loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# load weights into new model
loaded_model.load_weights("model2.h5")
#print("Loaded model from disk")
loaded_model.save('model2.hdf5')
model=load_model('model2.hdf5', compile=False)

gap_weights = model.layers[-1].get_weights()[0]
model  = Model(inputs=model.input,outputs=(model.layers[-3].output,model.layers[-1].output))




#features = final conv layer
#results = dense layer
features, results = model.predict(X_test)
bad = []
good = []
for x in range(len(X_test) - 1):
    pred = np.argmax(results[x])
    if results[x][pred] <= .50:
        bad.append([Y_test[x], x, pred, results[x][pred]])
    if results[x][pred] >= .90:
        good.append([Y_test[x], x, pred, results[x][pred]])

ones_good = [x for x in good if x[2] == 2]
ones_bad = [x for x in bad if x[2] == 2]
#print(ones_good)
print(ones_bad)

#print(X_test.shape)
#print(X_train.shape)
for x in ones_bad:
    X_train = np.append(X_train, np.swapaxes(X_test[x[1]], 0, 2), axis=0)
    Y_train = np.append(Y_train, Y_test[x[1]])



#features.shape
for idx in range(10):
    features_for_one_img = features[idx,:,:,:]
 
    #cam_features = sp.ndimage.zoom(features_for_one_img, (height_roomout, width_roomout, 1), order=2)
    #print(cam_features.shape)
    pred = np.argmax(results[idx])
    cam_features = features_for_one_img
    
    
    plt.figure(facecolor='white')
    cam_weights = gap_weights[:,pred]
    cam_output  = np.dot(cam_features,cam_weights)
    #print(features_for_one_img.shape)

    buf = 'Predicted Class = ' +str( pred )+ ', Probability = ' + str(results[idx][pred])

    plt.xlabel(buf)
    plt.imshow(np.squeeze(X_test[idx],-1), alpha=0.5)
    plt.imshow(cam_output, cmap='jet', alpha=0.5)

    plt.show()

    

