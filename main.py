import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten,BatchNormalization
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
from keras.backend import tensorflow_backend
import keras.callbacks
import keras.backend.tensorflow_backend as KTF
import tensorflow as tf
from sklearn.model_selection import train_test_split
from netCDF4 import Dataset
import sys
import os.path
import time
import argparse
import matplotlib.pyplot as plt
import math
from model_const import Network

#エポック,バッチサイズ指定
EPOCHS = 10000
BATCH = 32
#保存先ディレクトリを指定
common_name = '_FCN'
epochPath = '/result/epoch'+ common_name + '/'
modelPath = '/result/model'+ common_name + '/'
LCPath = '/result/LC'+ common_name + '/'

#ディレクトリがなければ生成
def if_not_exists_mkdir(dir_path):
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)

#学習過程をプロット(epochに対するloss)
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

#ラーニングカーブをプロット
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
    
    #コマンドライン引数読み取り
    args = sys.argv
    path = args[1]

    #netCDFファイル読み込み
    origin = Dataset(path,'r')
    trainX_orig = origin.variables['imgX'][:]
    trainY_orig = origin.variables['imgY'][:]
    
    #データフォーマット指定
    K.set_image_data_format("channels_first")
    val_loss = []
    train_loss = []
    dataset_size = []
    #トレーニング,テスト用に分割
    trainX, testX, trainY, testY = train_test_split(trainX_orig,trainX_orig,test_size = 0.3)
    
    #トレーニングデータ数を変化させてラーニングカーブ作成
    for train_size in range(trainsize ,1 ,-20):
        inputX = trainX[:train_size,:,:,:]
        inputY = trainY[:train_size,:,:,:]

        #モデル取得 
        channels = 1   
        model_ins = Network(channels,img_wigth,img_height)
        network = model_ins.get_model()
        optimizer = tf.train.AdamOptimizer(0.0005)
    
        network.compile(loss="mse", 
                        optimizer=optimizer, 
                        metrics=["mse"])
        #モデルの訓練
        history = network.fit(inputX,
                              inputY, 
                              epochs = EPOCHS, 
                              verbose = 1,
                              batch_size = BATCH,
                              validation_data = (testX, testY))
        
        #モデルの保存
        network.save(modelPath + 'ts'+str(train_size)+'.h5' , include_optimizer = False)
        
        print('which : ' + str(train_size) +' ---------------------------------------------------')
        
        plot_epochs(history,train_size)
        val_loss.append(history.history['val_mean_squared_error'][-1])
        train_loss.append(history.history['mean_squared_error'][-1])    
        dataset_size.append(train_size)

    plot_LC(dataset_size, train_loss, val_loss)
    
    end = time.time()
    print("time : {}[m]".format((end - start)/60))
       
