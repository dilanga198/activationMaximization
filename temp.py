import os
import numpy as np
import io

from PIL import Image
from keras.datasets import mnist
from matplotlib import pyplot as plt

def crop_image(img,tol=0):
    # img is 2D or 3D image data
    # tol  is tolerance
    mask = img>tol
    if img.ndim==3:
        mask = mask.all(2)
    m,n = mask.shape
    mask0,mask1 = mask.any(0),mask.any(1)
    col_start,col_end = mask0.argmax(),n-mask0[::-1].argmax()
    row_start,row_end = mask1.argmax(),m-mask1[::-1].argmax()
    return img[row_start:row_end,col_start:col_end]


(X_train,Y_train),(X_test,Y_test)  = mnist.load_data()

X_test = X_test.reshape((X_test.shape[0],X_test.shape[1],X_test.shape[2],1))
X_test  = X_test/255
X_test  = X_test.astype('float')

X_test = X_test[:3000,:,:,:]

data_dir = '/home/tseibel/Desktop/Debug_Program/data/'
classes = ['low/', 'high/','miss/','all/']

Missing_Activations = ['33']

class_dict = {}
activation_images = []
array2 = []
for npyfile in os.listdir(data_dir + 'low/'):
    array = np.load(data_dir + 'low/' + npyfile)
    image = np.array(X_test[int(npyfile[:-10])])
    count = 0
    for act in Missing_Activations:
        #print(np.amax(array[:,:,int(act)]))
        image2 = image.copy()
        x = array[:,:,int(act)] < (np.amax(array[:,:,int(act)])*.5)
        image2[x] = 0

        #Crops the images down to desired selection
        #image2 = crop_image(image2)
        plt.suptitle('Image: ' + str(npyfile[:-10]) + ' Filter: ' + str(act))
        
        #overlays both activation and underlying image
        #plt.imshow(image)
        plt.imshow(image2, alpha=.75)

        #Cropped Images down to the same size
        #plt.imshow(image2, extent=[0, 8.0, 0, 8.0])
        '''
        with io.BytesIO() as out:
            plt.savefig(out, format="png", dpi=80)  # Add dpi= to match your figsize
            pic = Image.open(out)
            pix = np.array(pic.getdata(), dtype=np.uint8).reshape(pic.size[1], pic.size[0], -1)
        array2.append(pix)
        '''
        activation_images.append(np.array(image2))
        plt.savefig('/home/tseibel/Desktop/Debug_Program/aug_activations/' + 'image' + str(npyfile[:-10]) + '.png')
        plt.show()


for image in activation_images:
    print(image.shape)

'''
for image in array2:
    plt.imshow(image)
    plt.show
'''
