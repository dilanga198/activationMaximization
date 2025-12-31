
'''
##########
PRINT CODE FROM run.py
##########
if value == 'Y' or value == 'y':

    files = [filename for filename in os.listdir(data_dir + 'all/')]
    for file in files:
        x = np.load(data_dir + 'all/' + file)
        r = []
        r.append(x[:,:,Missing_Activations[0]])
    r = np.array(r)
    all_mean = r.mean(axis=0)

    files = [filename for filename in os.listdir(data_dir + 'high/')]
    for file in files:
        x = np.load(data_dir + 'high/' + file)
        y = []
        y.append(x[:,:,Missing_Activations[0]])
    y = np.array(y)
    high_mean = y.mean(axis=0)
    #np.save('high_mean.npy', y)

    files = [filename for filename in os.listdir(data_dir + 'low/')]
    for file in files:
        x = np.load(data_dir + 'low/' + file)
        s = []
        s.append(x[:,:,Missing_Activations[0]])
    s = np.array(s)
    low_mean = s.mean(axis=0)
    zeros = low_mean < 0
    low_mean[zeros] = 0
    #np.save('low_mean.npy', s)

    files = [filename for filename in os.listdir(data_dir + 'miss/')]
    for file in files:
        x = np.load(data_dir + 'miss/' + file)
        #print(x.shape)
        z = []
        z.append(x[:,:,Missing_Activations[0]])
        #for key in Missing_Activations:
            #a_Map = x[:,:,key]
            #plt.title('low')
            #plt.imshow(a_Map, interpolation='nearest')
            #plt.show()
    z = np.array(z)
    miss_mean = z.mean(axis=0)
    #####################################
    #############PLOTS###################
    #####################################
    fig, axs = plt.subplots(2,2)
    fig.suptitle('High/Low AVG plots')
    axs[0,0].imshow(all_mean)
    axs[0,1].imshow(high_mean)
    axs[1,0].imshow(low_mean)
    axs[1,1].imshow(miss_mean)
    plt.show()
    #####################################
    #High - 35, Low - 456, Missed - 119
    diff_1 = high_mean - low_mean
    zeros = diff_1 < 0
    diff_1[zeros] = 0
    diff_2 = high_mean - miss_mean
    zeros = diff_2 < 0
    diff_2[zeros] = 0
    np.save('diff_mean.npy', diff_2)
    plt.suptitle('High/Low/Missed')
    plt.imshow(diff_2, 'Reds')
    #####################################
    diff_3 = low_mean - high_mean
    zeros = diff_3 < 0
    diff_3[zeros] = 0
    plt.imshow(diff_3, 'Greens', alpha = 0.5)
    #####################################
    diff_4 = miss_mean - high_mean
    zeros = diff_4 < 0
    diff_3[zeros] = 0
    plt.imshow(diff_4, 'Purples', alpha = 0.5)
    plt.show()
    #####################################
    #######OVERLAY PLOTS#################
    #####################################
    fig, axs = plt.subplots(2,2)
    fig.suptitle('High/Low/Miss AVG plots with Overlay')
    fig.suptitle('Sample Test Images with Overlay')
    axs[0,0].imshow(X_test[35])
    axs[0,0].imshow(diff_2, 'Reds', alpha=0.5)
    axs[0,0].imshow(diff_3, 'Greens', alpha=0.5)
    axs[0,1].imshow(X_test[456])
    axs[0,1].imshow(diff_2, 'Reds', alpha=0.5)
    axs[0,1].imshow(diff_3, 'Greens', alpha=0.5)
    axs[1,0].imshow(X_test[119])
    axs[1,0].imshow(diff_2, 'Reds', alpha=0.5)
    axs[1,0].imshow(diff_3, 'Greens', alpha=0.5)
    axs[1,1].imshow(diff_2, 'Reds')
    axs[1,1].imshow(diff_3, 'Greens', alpha=0.5)
    plt.show()
    #####################################
    fig, axs = plt.subplots(2,2)
    fig.suptitle('High/Low/Miss AVG plots with Overlay')
    fig.suptitle('Sample Test Images with Overlay')
    axs[0,0].imshow(X_test[35])
    axs[0,0].imshow(diff_2, 'Reds', alpha=0.5)
    axs[0,0].imshow(diff_3, 'Greens', alpha=0.5)
    axs[0,0].imshow(diff_4, 'Purples', alpha=0.5)
    axs[0,1].imshow(X_test[456])
    axs[0,1].imshow(diff_2, 'Reds', alpha=0.5)
    axs[0,1].imshow(diff_3, 'Greens', alpha=0.5)
    axs[0,1].imshow(diff_4, 'Purples', alpha=0.5)
    axs[1,0].imshow(X_test[119])
    axs[1,0].imshow(diff_2, 'Reds', alpha=0.5)
    axs[1,0].imshow(diff_3, 'Greens', alpha=0.5)
    axs[1,0].imshow(diff_4, 'Purples', alpha=0.5)
    axs[1,1].imshow(diff_2, 'Reds')
    axs[1,1].imshow(diff_3, 'Greens', alpha=0.5)
    axs[1,1].imshow(diff_4, 'Purples', alpha=0.5)
    plt.show()
'''