import os
import shutil
#import train_mnist as tm
import output_mnist as om
import load_mnist as lm
import run_SPM as rs
import itertools
import data_pull as dp
import math
import operator
import numpy as np

from PIL import Image
from keras.datasets import mnist
from matplotlib import pyplot as plt
from collections import Counter



(X_train,Y_train),(X_test,Y_test)  = mnist.load_data()

X_test = X_test.reshape((X_test.shape[0],X_test.shape[1],X_test.shape[2],1))
X_test  = X_test/255
X_test  = X_test.astype('float')

X_test = X_test[:1000,:,:,:]

#plt.imshow(X_test[35])
#plt.show()
#plt.imshow(X_test[1])
#plt.imshow(X_test[119], 'jet', alpha=0.5)
#plt.show()
#one = X_test[35]
#two = X_test[1]
#three = X_test[119]
#High - 35, Low - 456, Missed - 119


data_dir = '/data/'
#classes = ['low/', 'high/','miss/','all/']
classes = ['high/','miss/','all/']
labels = [0,1,2,3,4,5,6,7,8,9]

####################################
#########     TRAINING    ##########
####################################

value = input("Run Training? (Y/N)")
if value == 'Y' or value == 'y':
    os.system('python train_mnist.py')
else:
    print('No Training')

'''
#finds confidence across test labels
value = input("Run Confidence Test? (Y/N)")
if value == 'Y' or value == 'y':
    dp.delete_folders(data_dir,classes)
    #sampler - sample from each class
    #other_arrays - low, high, missed
    sampler, other_arrays = om.run()
else:
    print('No Confidence Test')
'''

####################################
#######    SPM Pairwise   ##########
####################################

'''
if len(other_arrays[0][0]) == 0 or len(other_arrays[1][0]) == 0 or len(other_arrays[2][0]) == 0:
    print('Not Enough Data to run Pair-Wise Comparison')
else:
    if value == 'Y' or value == 'y':
'''
    
all_selected_labels = {}

#user input into list - 4 5 -> ['4','5']
user_input = [str(x) for x in input("Run pair-wise comparison on which label? (0-9)").split()]

#comparison run seperately for each input label
for inputs in user_input:

    dp.delete_folders(data_dir,classes)

    #sampler - sample from each class
    #other_arrays - low, high, missed
    sampler, other_arrays = om.run()


    sampled = []
    for neurons in sampler[0]:
        sampled.append(neurons[1])

    lm.save_arrays(sampled, data_dir + sampler[1])
    
    for array in other_arrays:
        dp.sort_arrays(array[0], inputs, data_dir + array[1])

    all_freq = []
    frequencies = {}

    #Biulding Dictionaries of neuron frequency
    for c in classes:
        data_dir_list = os.listdir(data_dir + c)
        all_neurons = []
        for x,y in itertools.combinations(data_dir_list, 2):
            new_neurons = rs.run_SPM(x,y, data_dir+c)
            all_neurons += new_neurons
        print(c)
        frequency = dp.CountFrequency(all_neurons)
        all_freq.append(frequency)
        #frequency2 = {k:v for k, v in frequency.items()}

        # dictionary for the frequency of activations for most popular neurons
        frequencies[c[:-1]] = frequency
        #print(frequencies)
    all_selected_labels[inputs] = frequencies

print(all_selected_labels)





####################################
##########    TF-IDF    ############
####################################

max_high = {}
max_miss = {}

if len(all_selected_labels) > 0:
    tf_idf_value = input("Run tf-idf? (Y/N)")
    if tf_idf_value == 'Y' or tf_idf_value == 'y':
        for inputs in all_selected_labels:
            for key in inputs:
                frequencies = all_selected_labels[key]
                #print(key)
                high = frequencies['high']
                #low = frequencies['low']
                alll = frequencies['all'] 
                miss = frequencies['miss']


                #group = [high, low, alll, miss]
                group = [high, miss, alll]


                #### Build of all similarities found during pair-wise compairison for use during tf-idf calculations ###
                main = {}
                for dictionary in all_freq:
                    main = dp.mergeDict(main, dictionary)
                
                #print('main')
                #print(main)

                # Merge dictionaries and add values of common keys in a list
                #act_frequency = sum(all_freq[0].values()) + sum(all_freq[1].values()) + sum(all_freq[2].values()) + sum(all_freq[3].values())
                act_frequency = sum(all_freq[0].values()) + sum(all_freq[1].values()) + sum(all_freq[2].values())

                #print(group)

                #PRINT tf-idf values for designated classes
                
                #print('all_freq')
                #print(all_freq)
                #print('frequencies')
                #print(frequencies)
                #print('group')
                #print(group)
                tf_high = {}
                tf_miss = {}
                tf_all ={}
                for key in main:
                    count = 0
                    for g in group:
                        #if key in g and g[key] >= 145:
                        if key in g:
                            count +=1
                    #print(count)
                    #print('key: ', key)
                    for f in frequencies:

                        #if key in frequencies[f] and frequencies[f][key] >= 145:
                        if key in frequencies[f]:
                            #print(f)
                            #print(f, ' value: ' , (frequencies[f][key] / sum(frequencies[f].values()))  * np.log((3 / count) + 1))
                            if f == 'high':
                                tf_high[key] = (frequencies[f][key] / sum(frequencies[f].values()))  * np.log((3 / count) + 1)
                            if f == 'miss':
                                tf_miss[key] = (frequencies[f][key] / sum(frequencies[f].values()))  * np.log((3 / count) + 1)
                            if f == 'all':
                                tf_all[key] = (frequencies[f][key] / sum(frequencies[f].values()))  * np.log((3 / count) + 1)    
                
                #print(tf_high)
                #print(tf_miss)
                #print(tf_all)

                for key in tf_all:
                    if key in tf_miss:
                        if tf_all[key] * 1.2 > tf_miss[key]:
                            del tf_miss[key]
                    if key in tf_high:
                        if tf_all[key] * 1.2 > tf_high[key]:
                            del tf_high[key]
                
                act_frequency = sum(alll.values())

                Missing_Activations = []
                #print('inputs: ', inputs)
                max_list_high = []
                max_list_miss = []
                m_high = 0
                m_miss = 0

                #print('high')
                for key in tf_high:
                    #if key not in low.keys() and key not in miss.keys():
                    #print('high')
                    if key not in tf_miss:
                        #print(key, tf_high[key])
                        if tf_high[key] > m_high:
                            max_high[inputs] = [inputs, key, tf_high[key]]
                        max_list_high.append([inputs, key, tf_high[key]])

                    elif tf_high[key] > tf_miss[key]:
                        if tf_high[key] > m_high:
                            max_high[inputs] = [inputs, key, tf_high[key]]
                        #print(key, tf_high[key])
                        max_list_high.append([inputs, key, tf_high[key]])

                #print('miss')
                for key in tf_miss:
                    #print('miss')
                    if key not in tf_high:
                        #print(key, tf_miss[key])
                        if tf_miss[key] > m_miss:
                            max_miss[inputs] = [inputs, key, tf_miss[key]]
                        max_list_miss.append([inputs, key, tf_miss[key]])
                    elif tf_miss[key] > tf_miss[key]:
                        #print(key, tf_miss[key])
                        if tf_miss[key] > m_miss:
                            max_miss[inputs] = [inputs, key, tf_miss[key]]
                        max_list_miss.append([inputs, key, tf_miss[key]])


print('only prints max value')
print('{input class, [input class, tf_idf value]}')
print('max_high: ', max_high)
print('max_miss: ', max_miss)


























            #original comparison run on only one label 
'''
        if int(user_input) < 10:
            sampled = []
            for neurons in sampler[0]:
                print('neruons')
                print(neurons)
                sampled.append(neurons[1])
            lm.save_arrays(sampled, data_dir + sampler[1])
            for array in other_arrays:
                dp.sort_arrays(array[0], value, data_dir + array[1])
        else:
            print('Invalid Input')


    #Added the run-feature into prior code block    

    value = input("Run pair-wise comparison? (Y/N)")
    if value == 'Y' or value == 'y':
        all_freq = []
        frequencies = {}
        for c in classes:
            data_dir_list = os.listdir(data_dir + c)
            all_neurons = []
            for x,y in itertools.combinations(data_dir_list, 2):
                new_neurons = rs.run_SPM(x,y, data_dir+c)
                all_neurons += new_neurons
            print(c)
            frequency = dp.CountFrequency(all_neurons)
            #print(frequency)
            all_freq.append(frequency)
            frequency2 = {k:v for k, v in frequency.items()}
            #print(frequency)
            frequencies[c[:-1]] = frequency
            #print(frequencies)

high = frequencies['high']
low = frequencies['low']
alll = frequencies['all'] 
miss = frequencies['miss']

group = [high, low, alll, miss]


main = {}
for dictionary in all_freq:
    main = mergeDict(main, dictionary)
#print(main)

# Merge dictionaries and add values of common keys in a list
act_frequency = sum(all_freq[0].values()) + sum(all_freq[1].values()) + sum(all_freq[2].values()) + sum(all_freq[3].values())

print(group)
for key in main:
    count = 0
    for g in group:
        if key in g:
            count +=1
    print(count)
    print('key: ', key)
    for f in frequencies:
        if key in frequencies[f]:
            print(f, ' value: ' , (frequencies[f][key] / sum(frequencies[f].values()))  * np.log((4 / count)))
        else:
            print(f, ' value: ', 0)

for key in alll.keys():
    if key in low.keys():
        del low[key]
    if key in miss.keys():
        del miss[key]
    if key in high.keys():
        del high[key]

act_frequency = sum(alll.values())

Missing_Activations = []
for key in high.keys():
    if key not in low.keys() and key not in miss.keys():
        print(key, high[key])
        Missing_Activations.append(key)

Missing_Activations = [33, 75]

value = input("Plot Results? (Y/N)")
if value == 'Y' or value == 'y':
    print
'''