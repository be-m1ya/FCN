import numpy as np
from keras.layers import Dense, Dropout, Flatten,BatchNormalization,Input,Conv2D, MaxPooling2D,Activation
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
import keras.callbacks
import keras.backend.tensorflow_backend as KTF
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping


class Network(object):
    def __init__(self,channels,img_width,img_height):
        middle_activation = 'relu'
        output_activation = 'sigmoid'

        #Functional API
        inputs = Input(shape = (channels,img_width,img_height))

        #convolution
        x1 = Conv2D(64,(4,4),strides= (2,2),padding='same')(inputs)
        x1 = BatchNormalization()(x1)
        
        x2 = Activation(middle_activation)(x1)
        x2 = Conv2D(64,(4,4),strides= (2,2))(x2)
        
        #deconvolution
        x3 = Activation(middle_activation)(x2)
        x3 = Conv2DTranspose(64,(4,4),strides=(2,2))(x3)
        
        merged = merge([x1,x3],mode="sum")

        out = Activation(middle_activation)(merged)
        out = Conv2DTranspose(64,(4,4),strides=(2,2))(out)
        predictions = Activation(output_activation)(out)        
        
        self.model = Model(inputs=inputs, outputs=predictions)        

    def get_model(self):
        return self.model
