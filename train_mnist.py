# -*- coding: utf-8 -*-

# **MNIST - No pooling layers**
"""
Pooling looses the spacial mapping of features to the image - increases the training and predicting overhead
"""

import keras
import math
import numpy as np
import matplotlib.pyplot as plt
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

"""# **PREPARE THE DATA**"""

strategy = tf.distribute.MirroredStrategy()

#with strategy.scope():
(X_train,Y_train),(X_test,Y_test)  = mnist.load_data()


#X_train = np.load('X_train.npy', allow_pickle=True)
#Y_train = np.load('Y_train.npy', allow_pickle=True)

#print(X_train.shape)
#print(Y_train.shape)

X_train = X_train.reshape((X_train.shape[0],X_train.shape[1],X_train.shape[2],1))

#X_test = X_test.reshape((X_test.shape[0],X_test.shape[1],X_test.shape[2],1))

X_train = X_train/255
#X_test  = X_test/255
X_train = X_train.astype('float')
#X_test  = X_test.astype('float')
X_train[0].shape

X_train = X_train[:4000,:,:,:]
Y_train = Y_train[:4000]



"""# **BUILD THE NETWORK**"""
np.random.seed(0)
model = Sequential()
model.add(Conv2D(16,input_shape=(28,28,1),kernel_size=(3,3),activation='relu',padding='same'))
#model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(32,kernel_size=(3,3),activation='relu',padding='same'))
#model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64,kernel_size=(3,3),activation='relu',padding='same'))
#model.add(MaxPooling2D())

model.add(Conv2D(128,kernel_size=(3,3),activation='relu',padding='same'))
model.add(GlobalAveragePooling2D())
model.add(Dense(10,activation='softmax'))

model.summary()
model.compile(loss='sparse_categorical_crossentropy',metrics=['accuracy'],optimizer='adam')

model.fit(X_train,Y_train,batch_size=10,epochs=5,validation_split=0.1,shuffle=True)

model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

model.save_weights('model.h5')


