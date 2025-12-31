from __future__ import division

import PIL
from PIL import Image
import numpy as np
import data_pull as dp
import SPM_DVT as sd
import scipy.misc

jpg_dir_og = '/home/tseibel/Desktop/Research'
npy_dir_og = '/home/tseibel/Desktop/Research'

#takes layer and desired filters and expands them
def select_expand(layer, desired_layers, n):
    count = 0
    output = np.empty(shape=[n * layer.shape[1],n * layer.shape[1]])
    for act in layer:
        if count in desired_layers:
            #how much bigger we need to new array to be
            y = np.kron(act, np.ones((n,n),dtype=int))
            output = np.dstack((output, y))
        count +=1
    return output

#Adjust original image size if needed.
def shrink_jpg(jpg_array, output):
    while jpg_array[:,:,0].shape[0] > output.shape[1]:
        if output.shape[1] + 2 >= jpg_array[:,:,0].shape[0]:
            jpg_array = jpg_array[1:-1,1:-1]
        else:
            jpg_array = jpg_array[1:,:-1]
    return jpg_array

images = ['11_4famp.npy','22_4famp.npy', '54_4famp.npy','164_4famp.npy','217_4famp.npy']

for image in images:
    jpg_dir = jpg_dir_og + image
    npy_dir = npy_dir_og + image + '.npy'

    #Import Image
    #jpg_img = Image.open(jpg_dir)

    #turn jpg into 3D array (R/G/B)
    #jpg_array = np.asarray(jpg_img)

    #shape of one layer of array -->(x,x)
    jpg_shape = jpg_array[:,:,0].shape

    #activation layer filters --> (x,x,x)
    layer = dp.open(npy_dir,'')[:,:,:,0]

    #shape of one filter --> (x,x)
    act_shape = layer.shape[1]

    #how many times bigger we need to make the filter arrays
    n = jpg_shape[0]//act_shape

    #save path
    path = '/home/tseibel/deep-visualization-toolbox/data_pull/aug_images/'

    #for each activation in the layer
    all_desired_layers = [[25],[20],[171],[124],[76],[30],[15],[94],[133],[168],[102]]
    for desired_layers in all_desired_layers:
        select_acts = select_expand(layer, desired_layers, n)
        aug_jpg_array = shrink_jpg(jpg_array, select_acts)

        #adjust array dimensions for iteration
        select_acts = np.moveaxis(select_acts, -1, 0)
        aug_jpg_array = np.array(np.moveaxis(aug_jpg_array, -1, 0))

        #normalize select_acts against given threshold
        thresh = 0.001

        where_are_NaNs = np.isnan(select_acts)
        select_acts[where_are_NaNs] = 0

        select_acts = np.where(select_acts > thresh, 1, 0)


        #for every selected activation in a given layer
        #modify each layer of the 
        for x in range(0, select_acts.shape[0]):
            used_array = aug_jpg_array.copy()
            for y in range(0,used_array.shape[0]):
                #used_array = np.where(select_acts[y,:,:] == 0, 0, used_array)
                used_array[y,:,:][select_acts[x,:,:] < 1] = 0
            used_array = np.array(np.moveaxis(used_array, 0, -1))
            scipy.misc.imsave(path + str(desired_layers)[1:-1] + '_' + image +  '.jpg', used_array)
