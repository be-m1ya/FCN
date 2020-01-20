import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten,BatchNormalization
from keras.layers.convolutional import Conv2D, MaxPooling2D
import numpy as np
from keras.utils import np_utils
from keras import backend as K
import keras.callbacks
import keras.backend.tensorflow_backend as KTF
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
from netCDF4 import Dataset
import sys
import os.path
import time
import argparse
import matplotlib.pyplot as plt
import math
from keras.backend import tensorflow_backend
from model_const import Network

EPOCHS = 10000
BATCH = 32

common_name = '_FCN'
epochPath = '/result/epoch'+ common_name + '/'
modelPath = '/result/model'+ common_name + '/'
LCPath = '/result/LC'+ common_name + '/'

#if there's not dir, make dir
def if_not_exists_mkdir(dir_path):
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)

def plot_epochs(hisotry,train_size):
    epochs = range(len(history.history['mean_squared_error']))
    plt.figure()
    plt.xlabel('epochs')
    plt.ylabel('mean squared error')
    plt.plot( epochs , history.history['val_mean_squared_error'] ,label = 'val_loss')
    plt.plot( epochs , history.history['mean_squared_error'] ,label = 'train_loss')
    plt.ylim([0,0.15])
    plt.grid(which='major',color='black',linestyle='-')
    plt.legend()
    plt.savefig(epochPath +str(train_size)+'_epochs.png')


def plot_LC(dataset_size, t_loss, v_loss)
    plt.figure()
    plt.xlabel('traindata_size')
    plt.ylabel('mean squared error')
    plt.plot( dataset_size , v_loss ,label = 'val_loss')
    plt.plot( dataset_size , t_loss ,label = 'train_loss')
    plt.ylim([0,0.15])
    plt.grid(which='major',color='black',linestyle='-')
    plt.legend()
    plt.savefig(LCPath + 'LC.png')

if __name__ == "__main__":
    start = time.time()
    
    if_not_exists_mkdir(epochPath)
    if_not_exists_mkdir(modelPath)
    if_not_exists_mkdir(LCPath)

    args = sys.argv
    path = args[1]

    #load netCDF
    origin = Dataset(path,'r')
    trainX_orig = origin.variables['imgX'][:]
    trainY_orig = origin.variables['imgY'][:]
    
    #data format
    K.set_image_data_format("channels_first")
    #input channnels
    channels = 1
    val_loss = []
    train_loss = []
    dataset_size = []
    #split 
    trainX, testX, trainY, testY = train_test_split(trainX_orig,trainX_orig,test_size = 0.3)
    
    for train_size in range(trainsize ,1 ,-20):
        #get training set
        inputX = trainX[:train_size,:,:,:]
        inputY = trainY[:train_size,:,:,:]

        #get_model
        model_ins = Network(channels,img_wigth,img_height)
        network = model_ins.get_model()
        optimizer = tf.train.AdamOptimizer(0.000005)
    
        network.compile(loss="mse", 
                        optimizer=optimizer, 
                        metrics=["mse"])
    
        history = network.fit(trainX,
                              trainY, 
                              epochs = EPOCHS, 
                              verbose = 1,
                              batch_size = BATCH,
                              validation_data = (testX, testY))
    
        network.save(modelPath + 'ts'+str(train_size)+'.h5' , include_optimizer = False)
        
        print('which : ' + str(train_size) +' ---------------------------------------------------')
        
        plot_epochs(history,train_size)
        val_loss.append(history.history['val_mean_squared_error'][-1])
        train_loss.append(history.history['mean_squared_error'][-1])    
        dataset_size.append(train_size)

    plot_LC(dataset_size, train_loss, val_loss)
    
    end = time.time()
    print("time : {}[m]".format((end - start)/60))
       
