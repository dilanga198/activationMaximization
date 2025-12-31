# -*- coding: utf-8 -*-


import keras
import math
import os
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
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



def save_arrays(inputs, dir):
  (X_train,Y_train),(X_test,Y_test)  = mnist.load_data()

  #X_train = np.load('X_train.npy', allow_pickle=True)
  #Y_train = np.load('Y_train.npy', allow_pickle=True)
  X_train = X_train.reshape((X_train.shape[0],X_train.shape[1],X_train.shape[2],1))
  X_test = X_test.reshape((X_test.shape[0],X_test.shape[1],X_test.shape[2],1))
  #X_train = X_train/255
  X_test  = X_test/255
  #X_train = X_train.astype('float')
  X_test  = X_test.astype('float')



  json_file = open('model.json', 'r')

  loaded_model_json = json_file.read()
  json_file.close()
  loaded_model = model_from_json(loaded_model_json)

  # load weights into new model
  loaded_model.load_weights("model.h5")
  print("Loaded model from disk")

  loaded_model.save('model.hdf5')
  model=load_model('model.hdf5', compile=False)
  


  #for layer in model.layers:
  #  print(layer)
    
  ixs = [0, 1, 2, 3]  
  outputs = [model.layers[i].output for i in ixs]
  the_model = Model(inputs=model.inputs, outputs=outputs)

  #inputs = [43, 149, 221, 298, 321, 326, 612, 646, 741, 926, 35, 82, 106, 147, 186, 199, 236, 237, 256, 258, 10, 2, 35, 30, 4, 1, 21, 0, 84, 9]
  for input in inputs:
    img = X_test[input]
    # convert the image to an array
    img = img_to_array(img)
    # expand dimensions so that it represents a single 'sample'
    img = expand_dims(img, axis=0)
    # get feature map for first hidden layer
    feature_maps = the_model.predict(img)

    count  = 1
    for fmap in feature_maps:
      #print(fmap.shape)
      if count == 4:
        if 20 > len([name for name in os.listdir(dir)]):
          np.save(dir + str(input) + '_' + str(count) + 'fmap.npy',fmap[0,:,:,:])
          square = int(math.sqrt(fmap.shape[3]+1))
          # plot all 64 maps in an 8x8 squares
          ix = 1
          #layer visualization
          '''
          pyplot.figure(figsize = (12,12))
          pyplot.suptitle('Block '+str(count), fontsize=36)
          for _ in range(square):
            for _ in range(square):
              if ix > fmap.shape[3]:
                continue
              # specify subplot and turn of axis
              ax = pyplot.subplot(square, square, ix)
              ax.set_xticks([])
              ax.set_yticks([])
              # plot filter channel in grayscale
              pyplot.imshow(fmap[0,:, :, ix-1], cmap='gray')
              ix += 1
          # show the figure
          pyplot.show()
          '''
      count = count+1
