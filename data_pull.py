from __future__ import division

import numpy as np
import xlsxwriter
import os
import shutil
from heapq import nlargest
import math
import operator
import load_mnist as lm

#save any array to .npy file
def save(name, array, dir):
    np.save(dir + name, array)

#open any array saved to .npy file
def open(name, dir):
    return np.load(dir + name)

#takes 3D array and converts it to 2D
def to_array_2D(array_3D):
    array_2D = np.reshape(array_3D, (array_3D.shape[0] * array_3D.shape[1] * array_3D.shape[2], array_3D.shape[3]))
    return array_2D

#takes 2D array and converts it to excell for better viewing
def to_excel(array_2D, orginal_shape, name, dir):
    workbook = xlsxwriter.Workbook(dir + name)
    worksheet = workbook.add_worksheet()

    row = 0
    count = 0
    index = 0

    for col, data in enumerate(array_2D):
        worksheet.write_column(row,col - index,data)
        count += 1
        if count == orginal_shape[1]:
            row += 3
            count = 0
            index += orginal_shape[1]

    workbook.close()

def Average(array):
    return sum(array) / len(array)

#takes 2D array in .npy form and converts it to dictionary
def to_Dict(array_2D, shape):
    all_avg_active = Average(array_2D)
    filter_size = shape[1] * shape[2]
    my_dict = {}
    index = 1
    total_activation = 0
    num_Active = 0
    print(shape)
    print(array_2D.shape)
    for activation in array_2D:
        #checks avg of activation values
        if activation[0] > all_avg_active[0] + .15:
            num_Active += 1
            total_activation += activation[0]
        #checks % of neurons activated
        if index % filter_size == 0 and num_Active/filter_size >= .20:
            #[Average value fo activations, total activation across filter, % of activations, number of activatation on filter]
            dict[(index/filter_size) - 1] = [total_activation/num_Active, total_activation, num_Active/filter_size, num_Active]
            total_activation = 0
            num_Active = 0
        index += 1
    return my_dict

#avg Activation
#numer of Activations

#compares to dicts of activations
def comp_Dicts(dict1, dict2, shape_3D):
    comp_dict = {}
    for key in dict1.keys():
        if key in dict2.keys():
            print(key)

def k_largest(N, final_dict,to_norm):
    res = nlargest(N, final_dict, key = final_dict.get) 
    sum_filters = 0
    for r in res:
        sum_filters += final_dict[r][0]
    return res

def quadrants(array):
    a, b, c, d = array[:len(array)//2, :len(array)//2], array[len(array)//2:, :len(array)//2], array[:len(array)//2, len(array)//2:], array[len(array)//2:, len(array)//2:]
    return a, b, c, d

def max_quad(quads):
    max = 0
    count = 0
    name = None
    for quad in quads:
        if quad.sum() > max:
            max = quad.sum()
            name = count
        count += 1
    return max, name

def CountFrequency(my_list): 

    # Creating an empty dictionary  
    freq = {} 
    for item in my_list: 
        if (item in freq): 
            freq[item] += 1
        else: 
            freq[item] = 1
    '''
    x = input('Normalize Frequency? (Y/N)')
    if x == 'Y' or x == 'y':
        freq = dict_norm(freq)
    '''
    list_of_values = []
    for key, value in freq.items(): 
        list_of_values.append(value)
        print ("% d : % d"%(key, value)) 
    return freq

def delete_folders(folders, classes):
    for aclass in classes:
        folder = folders + aclass
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))

'''
def dict_norm(d):
    factor=1.0/sum(float(str(d.values())))
    for k in d:
        d[k] = float(str(d[k]))*factor
    return d
'''

def dict_norm(d):
    factor=1.0/math.fsum(d.values())
    for k in d:
        d[k] = d[k]*factor
    key_for_max = max(d.items(), key=operator.itemgetter(1))[0]
    diff = 1.0 - math.fsum(d.values())
    #print "discrepancy = " + str(diff)
    d[key_for_max] += diff
    return d

def sort_arrays(the_array, value, dir):
    names = []
    #print(the_array[int(value)])
    for neurons in the_array[int(value)]:
        names.append(neurons[1])
    lm.save_arrays(names, dir)

#Merge dictionaries and keep values of common keys in list
def mergeDict(dict1, dict2):

   dict3 = {**dict1, **dict2}
   for key, value in dict3.items():
       if key in dict1 and key in dict2:
               dict3[key] = value + dict1[key]
   return dict3