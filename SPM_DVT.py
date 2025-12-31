from __future__ import division

import itertools
import csv
from PIL import Image
import numpy as np
from os import listdir
from os.path import isfile, join
from numpy.lib.stride_tricks import as_strided
import data_pull as dp
from heapq import nlargest

def pool2d(A, kernel_size, stride, padding):
    '''
    2D Pooling

    Parameters:
        A: input 2D array
        kernel_size: int, the size of the window
        stride: int, the stride of the window
        padding: int, implicit zero paddings on both sides of the input
        pool_mode: string, 'max' or 'avg'
    '''
    # Padding
    A = np.pad(A, padding, mode='constant')

    # Window view of A
    output_shape = ((A.shape[0] - kernel_size)//stride + 1,
                    (A.shape[1] - kernel_size)//stride + 1)
    kernel_size = (kernel_size, kernel_size)
    A_w = as_strided(A, shape = output_shape + kernel_size, 
                        strides = (stride*A.strides[0],
                                   stride*A.strides[1]) + A.strides)
    A_w = A_w.reshape(-1, *kernel_size)

    # Return the result of pooling
    array = A_w.mean(axis=(1,2)).reshape(output_shape)
    return A_w.mean(axis=(1,2)).reshape(output_shape)

#input IMAGE1, IMAGE2, DEGREE
def pyr_match(OG_array_1, OG_array_2, degree):
    #applies SPM using modified pooling layer
    pym_vals = []
    total = 1
    count = 1
    the_max = []
    while count <= degree: 
        new_1_array = pool2d(OG_array_1, total, total, 0)
        new_2_array = pool2d(OG_array_2, total, total, 0)
        the_max.append(total ** 2)
        count += 1
        #Where More than Half the Squares are 0
        #with open('Input2.txt') as a:
        #    thresh_val = float(a.readline())
        thresh_val = 100
        condition_1_1 = (new_1_array >= thresh_val)
        condition_1_2 = (new_2_array >= thresh_val)
        #Where Half or More squares are 255
        #condition_2_1 = (new_1_array < thresh_val)
        #condition_2_2 = (new_2_array < thresh_val)
        part_1 = np.where(condition_1_1 & condition_1_2)
        #part_2 = np.where(condition_2_1 & condition_2_2)
        #pym_vals.append(len(part_1[0]) + len(part_2[0]))
        #print len(part_1[0])
        pym_vals.append(len(part_1[0]))
        #total = 2 ** count
        total = total * 2
    return pym_vals, the_max

#takes arrays we will compare the SPM and the degree
def pyr_all(array_1, array_2, degree):
    all_dict = {}
    diff_all_dict = {}
    layer_size = array_1.shape[0] - 1
    count = 0
    thresh = 1
    answer = 0
    array_1[array_1 > thresh] = 512
    array_2[array_2 > thresh] = 512
    array_1[array_1 <= thresh] = 0
    array_2[array_2 <= thresh] = 0
    while count <= layer_size:
        filter_1 = array_1[count][:,:]
        filter_2 = array_2[count][:,:]
        filter_size = filter_1.shape[0] * filter_1.shape[1]
        filter_zeros_1 = len(np.where(filter_1 == 0)[0])
        filter_zeros_2 = len(np.where(filter_2 == 0)[0])
        #with open('Input1.txt') as a:
        #    thresh_val = float(a.readline())
        thresh_val = .8
        #thresh_val = float(input("threshold? base value is .4"))
        if (filter_zeros_1 / filter_size) > thresh_val or (filter_zeros_2 / filter_size) > thresh_val:
            #print 'bad filters'
            xxxx = 1
        else:
            #print 'good filters'
            #print 'degree: ', degree
            answer, max = pyr_match(filter_1, filter_2, degree)
            #print 'answer: ', answer
            #print 'max: ', max
            all_dict[count], diff_all_dict[count] = the_alg(answer)
        count += 1


    return all_dict, diff_all_dict
                    

#takes an list of the SPM outputs from each level
#generates the final value for each level
def the_alg(output):
    #print output
    answer = []
    diff_answer =[]
    L = len(output) - 1
    index = 0
    out = 0
    out2 = 0
    diff_out = 0
    for x in output:
        #print 'x: ', x
        diff_value = ((1 / 2) ** ( index ) )  * (1 - ( x / ( ( 2 ** ( L - index ) ) ** 2 ) ) )
        value =  ((1 / 2) ** ( index ) )  * ( x / ( ( 2 ** ( L - index ) ) ** 2 ) )
        value2 =  ((1 / 2) ** ( index ) ) 
        #value =  ((1 / 2) ** ( L - index ) )  * ( x / ( 2 ** ( 2 * ( L - index ) ) ) )
        #value = ( 1 / ( 2 ** ( L - index ) ) ) * ( x / ( 2 ** ( 2 * ( L - index ) ) ) )
        #print value
        #print value2
        out += value
        out2 += value2
        diff_out += diff_value
        index +=1
    #print 'out:', out
    #print 'max: ', out2
    answer.append(out)
    diff_answer.append(diff_out)
    #print answer, diff_answer
    return answer, diff_answer
    #print answer[0]/answer[1]


def resize_array(array):
    array = np.transpose(array, (2, 1, 0))
    new_array = []
    count = 0
    div_list = [1,2,4,8,16,32,64,128,256,512,1024, 2048]
    max = array.shape[0]
    while count < max:
        layer = array[count].repeat(3, axis=0).repeat(3, axis=1)
        while layer.shape[0] not in div_list:
            if layer.shape[0] % 2 == 0:
                layer = layer[1:-1,1:-1]
            else:
                layer = layer[1:,:-1]
        count += 1
        new_array.append(layer)
    return np.array(new_array)

def degree(array):
    degree = 1
    while 2 ** degree != array[0].shape[0]:
        degree+=1 
    return degree + 1

def k_largest(N, final_dict,to_norm):
    res = nlargest(N, final_dict, key = final_dict.get) 
    sum_filters = 0
    values = []
    for r in res:
        sum_filters += final_dict[r][0]
        values.append([r, final_dict[r][0]])
    return res