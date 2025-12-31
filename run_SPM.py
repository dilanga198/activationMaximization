from __future__ import division
from PIL import Image
from os import listdir
from os.path import isfile, join
from numpy.lib.stride_tricks import as_strided
from heapq import nlargest

import SPM_DVT as sd
import itertools
import csv
import numpy as np
import data_pull as dp



def run_SPM(filter1, filter2, dir):
    #directory
    #array augmentation
    array_1 = sd.resize_array(np.load(dir + filter1))
    array_2 = sd.resize_array(np.load(dir + filter2))

    #print(array_1.shape)

    #degree
    degree = sd.degree(array_1)
    max_1 = np.amax(array_1)
    max_2 = np.amax(array_2)
    v1 = 512/max_1
    v2 = 512/max_2
    array_1 = v1 * array_1
    array_2 = v2 * array_2

    answers_dict, diff_dict = sd.pyr_all(array_1, array_2, degree)

    final_dict = {}
    for key in answers_dict:
        if answers_dict[key] > [0]:
            final_dict[key] = answers_dict[key]

    #print(final_dict)
    return dp.k_largest(40, final_dict,'y')

