# -*- coding: utf-8 -*-


import keras
import math
import numpy as np
import matplotlib.pyplot as plt
import keras.backend as K
import tensorflow as tf
import logging
import os

from keras.datasets import mnist
from keras.models import Sequential,Model
from keras.layers import Dense,Conv2D,Flatten,MaxPooling2D,GlobalAveragePooling2D
from keras.utils import plot_model
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import Model
from matplotlib import pyplot as plt
from numpy import expand_dims
from skimage import io
from keras.models import model_from_json
from keras.models import load_model


def run():
    #load in test set
    (X_train,Y_train),(X_test,Y_test)  = mnist.load_data()
    X_test = X_test.reshape((X_test.shape[0],X_test.shape[1],X_test.shape[2],1))
    X_test  = X_test/255
    X_test  = X_test.astype('float')
    X_test = X_test[:5000,:,:,:]

    #load in trained model
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)

    # load weights and add to trained
    loaded_model.load_weights("model.h5")
    loaded_model.save('model.hdf5')
    model=load_model('model.hdf5', compile=False)
    #gap_weights = model.layers[-1].get_weights()[0]
    model  = Model(inputs=model.input,outputs=(model.layers[-3].output,model.layers[-1].output))

    #features = final conv layer
    #results = dense layer
    features, results = model.predict(X_test)
    low = []
    high = []
    miss = []
    lowDict = {}
    for x in range(len(X_test) - 1):
        pred = np.argmax(results[x])
        if results[x][pred] < .50 and Y_test[x] == pred:
            low.append([Y_test[x], x, pred, results[x][pred]])
        if results[x][pred] >= .75 and Y_test[x] == pred:
            high.append([Y_test[x], x, pred, results[x][pred]])
        if Y_test[x] != pred:
            #print(X_test[x])
            miss.append([Y_test[x], x, pred, results[x][pred]])

    #print(low)
    labels = [0,1,2,3,4,5,6,7,8,9]

    low_sorted = []
    high_sorted = []
    miss_sorted = []
    count = 0
    w = []
    for label in labels:
        x = []
        y = []
        z = []
        for h in high:
            if h[0] == label:
                count += 1
                w.append(h)
            if count == 2:
                count = 0
                break
        for m in miss:
            if m[0] == label:
                x.append(m)
        for h in high:
            if h[0] == label:
                y.append(h)
        for l in low:
            if l[0] == label:
                z.append(l)
        miss_sorted.append(x)
        high_sorted.append(y)
        low_sorted.append(z)

    x = 0
    '''
    for lows in low_sorted:
        print('low conf for class', x, ':')
        print(len(lows)/len(X_test) * 100,'%')
        x += 1
    '''
    #high_conf = [x for x in high if x[0] == 2]
    #low_conf = [x for x in low if x[0] == 2]
    #miss_class = [x for x in miss if x[0] == 2]


    #print('miss')
    #print(miss_class)
    #print(miss)
    #print('low')
    #print(low_conf)
    #print('high')
    #print(high)
    #print(high_conf)


    #for x in miss_class:
    #    X_train = np.append(X_train, np.swapaxes(X_test[x[1]], 0, 2), axis=0)
    #    Y_train = np.append(Y_train, Y_test[x[1]])


    #np.save('X_train.npy', X_train)
    #np.save('Y_train.npy', Y_train)
    return [w,'all/'], [[low_sorted,'low/'], [high_sorted,'high/'], [miss_sorted, 'miss/']]