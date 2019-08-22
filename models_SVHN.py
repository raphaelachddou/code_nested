import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.datasets import mnist
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Input, Concatenate, Lambda, Softmax, Activation, BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Cropping2D, GlobalAveragePooling2D
from tensorflow.keras import backend as K
from time import time




def coarse_modelUnet(input_shape=(32, 32, 3)):
    input_img = Input(input_shape,name = 'input_1')
    c1 = (Conv2D(32, kernel_size=3, input_shape=input_shape, padding="same",name = 'conv2d'))(input_img)
    c1 = (BatchNormalization(name= 'batch_normalization' ))(c1)
    c1 = (Activation('relu',name = 'activation'))(c1)
    c2 = (Conv2D(32, 3, padding="same", name = 'conv2d_1'))(c1)
    c2 = (BatchNormalization(name= 'batch_normalization_1'))(c2)
    c2 = (Activation('relu',name = 'activation_1'))(c2)
    m1 = (MaxPooling2D(pool_size=2, name = 'max_pooling2d'))(c2)
    d1 = (Dropout(0.3, name = 'dropout'))(m1)

    c3 = (Conv2D(64, 3,padding="same",name = 'conv2d_2'))(d1)
    c3 = (BatchNormalization(name= 'batch_normalization_2'))(c3)
    c3 = (Activation('relu',name = 'activation_2'))(c3)
    c4 = (Conv2D(64, 3, padding="same",name = 'conv2d_3'))(c3)
    c4 = (BatchNormalization(name= 'batch_normalization_3'))(c4)
    c4 = (Activation('relu',name = 'activation_3'))(c4)
    m2 = (MaxPooling2D(pool_size=2, name = 'max_pooling2d_1'))(c4)
    d2 = (Dropout(0.3, name = 'dropout_1'))(m2)

    c5 = (Conv2D(128, 3,padding="same",name = 'conv2d_4'))(d2)
    c5 = (BatchNormalization(name= 'batch_normalization_4'))(c5)
    c5 = (Activation('relu',name = 'activation_4'))(c5)
    c6 = (Conv2D(128, 3, padding="same",name = 'conv2d_5'))(c5)
    c6 = (BatchNormalization(name= 'batch_normalization_5'))(c6)
    c6 = (Activation('relu',name = 'activation_5'))(c6)
    m3 = (MaxPooling2D(pool_size=2, name = 'max_pooling2d_2'))(c6)
    d3 = (Dropout(0.3, name = 'dropout_2'))(m3)
    c7 = (Conv2D(128, 3,padding="same",name = 'conv2d_6'))(d3)
    c7 = (BatchNormalization(name= 'batch_normalization_6'))(c7)
    c7 = (Activation('relu',name = 'activation_6'))(c7)

    f1 = (Flatten())(c7)
    res1 = (Dense(100, activation='relu',name = 'dense'))(f1)
    res1 = (Dropout(0.3, name = 'dropout_3'))(res1)
    res1 = (Dense(2, activation='softmax', name = 'coarse'))(res1)
    model = Model(input_img, res1)
    return(model)



def middle_modelUnet(input_shape=(32, 32, 3)):
    input_img = Input(input_shape,name = 'input_1')
    c1 = (Conv2D(32, kernel_size=3, input_shape=input_shape, padding="same",name = 'conv2d'))(input_img)
    c1 = (BatchNormalization(name= 'batch_normalization' ))(c1)
    c1 = (Activation('relu',name = 'activation'))(c1)
    c2 = (Conv2D(32, 3, padding="same", name = 'conv2d_1'))(c1)
    c2 = (BatchNormalization(name= 'batch_normalization_1'))(c2)
    c2 = (Activation('relu',name = 'activation_1'))(c2)
    m1 = (MaxPooling2D(pool_size=2, name = 'max_pooling2d'))(c2)
    d1 = (Dropout(0.3, name = 'dropout'))(m1)

    c3 = (Conv2D(64, 3,padding="same",name = 'conv2d_2'))(d1)
    c3 = (BatchNormalization(name= 'batch_normalization_2'))(c3)
    c3 = (Activation('relu',name = 'activation_2'))(c3)
    c4 = (Conv2D(64, 3, padding="same",name = 'conv2d_3'))(c3)
    c4 = (BatchNormalization(name= 'batch_normalization_3'))(c4)
    c4 = (Activation('relu',name = 'activation_3'))(c4)
    m2 = (MaxPooling2D(pool_size=2, name = 'max_pooling2d_1'))(c4)
    d2 = (Dropout(0.3, name = 'dropout_1'))(m2)

    c5 = (Conv2D(128, 3,padding="same",name = 'conv2d_4'))(d2)
    c5 = (BatchNormalization(name= 'batch_normalization_4'))(c5)
    c5 = (Activation('relu',name = 'activation_4'))(c5)
    c6 = (Conv2D(128, 3, padding="same",name = 'conv2d_5'))(c5)
    c6 = (BatchNormalization(name= 'batch_normalization_5'))(c6)
    c6 = (Activation('relu',name = 'activation_5'))(c6)
    m3 = (MaxPooling2D(pool_size=2, name = 'max_pooling2d_2'))(c6)
    d3 = (Dropout(0.3, name = 'dropout_2'))(m3)
    c7 = (Conv2D(128, 3,padding="same",name = 'conv2d_6'))(d3)
    c7 = (BatchNormalization(name= 'batch_normalization_6'))(c7)
    c7 = (Activation('relu',name = 'activation_6'))(c7)

    c8 = UpSampling2D(size=(2, 2),name = 'up_sampling2d')(c7)
    c9 = Concatenate(name = 'concatenate_1', axis = 3)([c6,c8])
    c9 = Conv2D(64, (3, 3),padding='same',name = 'conv2d_7')(c9)
    c9 = (BatchNormalization(name= 'batch_normalization_7'))(c9)
    c9 = (Activation('relu',name = 'activation_7'))(c9)

    f1 = (Flatten())(c7)
    res1 = (Dense(100, activation='relu',name = 'dense'))(f1)
    res1 = (Dropout(0.3, name = 'dropout_3'))(res1)
    res1 = (Dense(2, activation='softmax', name = 'coarse'))(res1)

    f2 = (Flatten())(c9)
    res2 = (Dense(50, activation='relu',name = 'dense_1'))(f2)
    res2 = (Dropout(0.3, name = 'dropout_4'))(res2)
    res2 = (Dense(4, activation='softmax', name = 'middle'))(res2)

    model = Model(input_img, [res1,res2])
    return(model)

def fine_modelUnet(input_shape=(32, 32, 3)):
    input_img = Input(input_shape,name = 'input_1')
    c1 = (Conv2D(32, kernel_size=3, input_shape=input_shape, padding="same",name = 'conv2d'))(input_img)
    c1 = (BatchNormalization(name= 'batch_normalization' ))(c1)
    c1 = (Activation('relu',name = 'activation'))(c1)
    c2 = (Conv2D(32, 3, padding="same", name = 'conv2d_1'))(c1)
    c2 = (BatchNormalization(name= 'batch_normalization_1'))(c2)
    c2 = (Activation('relu',name = 'activation_1'))(c2)
    m1 = (MaxPooling2D(pool_size=2, name = 'max_pooling2d'))(c2)
    d1 = (Dropout(0.3, name = 'dropout'))(m1)

    c3 = (Conv2D(64, 3,padding="same",name = 'conv2d_2'))(d1)
    c3 = (BatchNormalization(name= 'batch_normalization_2'))(c3)
    c3 = (Activation('relu',name = 'activation_2'))(c3)
    c4 = (Conv2D(64, 3, padding="same",name = 'conv2d_3'))(c3)
    c4 = (BatchNormalization(name= 'batch_normalization_3'))(c4)
    c4 = (Activation('relu',name = 'activation_3'))(c4)
    m2 = (MaxPooling2D(pool_size=2, name = 'max_pooling2d_1'))(c4)
    d2 = (Dropout(0.3, name = 'dropout_1'))(m2)

    c5 = (Conv2D(128, 3,padding="same",name = 'conv2d_4'))(d2)
    c5 = (BatchNormalization(name= 'batch_normalization_4'))(c5)
    c5 = (Activation('relu',name = 'activation_4'))(c5)
    c6 = (Conv2D(128, 3, padding="same",name = 'conv2d_5'))(c5)
    c6 = (BatchNormalization(name= 'batch_normalization_5'))(c6)
    c6 = (Activation('relu',name = 'activation_5'))(c6)
    m3 = (MaxPooling2D(pool_size=2, name = 'max_pooling2d_2'))(c6)
    d3 = (Dropout(0.3, name = 'dropout_2'))(m3)
    c7 = (Conv2D(128, 3,padding="same",name = 'conv2d_6'))(d3)
    c7 = (BatchNormalization(name= 'batch_normalization_6'))(c7)
    c7 = (Activation('relu',name = 'activation_6'))(c7)

    c8 = UpSampling2D(size=(2, 2),name = 'up_sampling2d')(c7)
    c9 = Concatenate(name = 'concatenate_1', axis = 3)([c6,c8])
    c9 = Conv2D(64, (3, 3),padding='same',name = 'conv2d_7')(c9)
    c9 = (BatchNormalization(name= 'batch_normalization_7'))(c9)
    c9 = (Activation('relu',name = 'activation_7'))(c9)


    c10 = UpSampling2D(size=(2, 2),name = 'up_sampling2d_1')(c9)
    c10 = Concatenate(name = 'concatenate_2', axis = 3)([c4,c10])
    c10 = Conv2D(32, (3, 3),padding='same',name = 'conv2d_8')(c10)
    c10 = (BatchNormalization(name= 'batch_normalization_8'))(c10)
    c10 = (Activation('relu',name = 'activation_8'))(c10)

    f1 = (Flatten())(c7)
    res1 = (Dense(100, activation='relu',name = 'dense'))(f1)
    res1 = (Dropout(0.3, name = 'dropout_3'))(res1)
    res1 = (Dense(2, activation='softmax', name = 'coarse'))(res1)

    f2 = (Flatten())(c9)
    res2 = (Dense(50, activation='relu',name = 'dense_1'))(f2)
    res2 = (Dropout(0.3, name = 'dropout_4'))(res2)
    res2 = (Dense(4, activation='softmax', name = 'middle'))(res2)

    f3 = (Flatten())(c10)
    res3 = (Dense(50, activation='relu',name = 'dense_2'))(f3)
    res3 = (Dropout(0.3, name = 'dropout_5'))(res3)
    res3 = (Dense(10, activation='softmax', name = 'fine'))(res3)



    model = Model(input_img, [res1,res2,res3])
    return(model)

def fine_modelUnet_single(input_shape=(32, 32, 3)):
    input_img = Input(input_shape,name = 'input_1')
    c1 = (Conv2D(32, kernel_size=3, input_shape=input_shape, padding="same",name = 'conv2d'))(input_img)
    c1 = (BatchNormalization(name= 'batch_normalization' ))(c1)
    c1 = (Activation('relu',name = 'activation'))(c1)
    c2 = (Conv2D(32, 3, padding="same", name = 'conv2d_1'))(c1)
    c2 = (BatchNormalization(name= 'batch_normalization_1'))(c2)
    c2 = (Activation('relu',name = 'activation_1'))(c2)
    m1 = (MaxPooling2D(pool_size=2, name = 'max_pooling2d'))(c2)
    d1 = (Dropout(0.3, name = 'dropout'))(m1)

    c3 = (Conv2D(64, 3,padding="same",name = 'conv2d_2'))(d1)
    c3 = (BatchNormalization(name= 'batch_normalization_2'))(c3)
    c3 = (Activation('relu',name = 'activation_2'))(c3)
    c4 = (Conv2D(64, 3, padding="same",name = 'conv2d_3'))(c3)
    c4 = (BatchNormalization(name= 'batch_normalization_3'))(c4)
    c4 = (Activation('relu',name = 'activation_3'))(c4)
    m2 = (MaxPooling2D(pool_size=2, name = 'max_pooling2d_1'))(c4)
    d2 = (Dropout(0.3, name = 'dropout_1'))(m2)

    c5 = (Conv2D(128, 3,padding="same",name = 'conv2d_4'))(d2)
    c5 = (BatchNormalization(name= 'batch_normalization_4'))(c5)
    c5 = (Activation('relu',name = 'activation_4'))(c5)
    c6 = (Conv2D(128, 3, padding="same",name = 'conv2d_5'))(c5)
    c6 = (BatchNormalization(name= 'batch_normalization_5'))(c6)
    c6 = (Activation('relu',name = 'activation_5'))(c6)
    m3 = (MaxPooling2D(pool_size=2, name = 'max_pooling2d_2'))(c6)
    d3 = (Dropout(0.3, name = 'dropout_2'))(m3)
    c7 = (Conv2D(128, 3,padding="same",name = 'conv2d_6'))(d3)
    c7 = (BatchNormalization(name= 'batch_normalization_6'))(c7)
    c7 = (Activation('relu',name = 'activation_6'))(c7)

    c8 = UpSampling2D(size=(2, 2),name = 'up_sampling2d')(c7)
    c9 = Concatenate(name = 'concatenate_1', axis = 3)([c6,c8])
    c9 = Conv2D(64, (3, 3),padding='same',name = 'conv2d_7')(c9)
    c9 = (BatchNormalization(name= 'batch_normalization_7'))(c9)
    c9 = (Activation('relu',name = 'activation_7'))(c9)


    c10 = UpSampling2D(size=(2, 2),name = 'up_sampling2d_1')(c9)
    c10 = Concatenate(name = 'concatenate_2', axis = 3)([c4,c10])
    c10 = Conv2D(32, (3, 3),padding='same',name = 'conv2d_8')(c10)
    c10 = (BatchNormalization(name= 'batch_normalization_8'))(c10)
    c10 = (Activation('relu',name = 'activation_8'))(c10)

    f3 = (Flatten())(c10)
    res3 = (Dense(50, activation='relu',name = 'dense_2'))(f3)
    res3 = (Dropout(0.3, name = 'dropout_5'))(res3)
    res3 = (Dense(10, activation='softmax', name = 'fine'))(res3)



    model = Model(input_img, [res3])
    return(model)



def coarse_modelBottleneck( input_shape=(32, 32, 3)):
    input_img = Input(input_shape,name = 'input_1')
    c1 = (Conv2D(32, kernel_size=3, input_shape=input_shape, padding="same",name = 'conv2d'))(input_img)
    c1 = (BatchNormalization(name= 'batch_normalization' ))(c1)
    c1 = (Activation('relu',name = 'activation'))(c1)
    c2 = (Conv2D(64, 3, padding="same", name = 'conv2d_1'))(c1)
    c2 = (BatchNormalization(name= 'batch_normalization_1'))(c2)
    c2 = (Activation('relu',name = 'activation_1'))(c2)
    m1 = (MaxPooling2D(pool_size=2, name = 'max_pooling2d'))(c2)
    d1 = (Dropout(0.3, name = 'dropout'))(m1)

    c3 = (Conv2D(64, 3,padding="same",name = 'conv2d_2'))(d1)
    c3 = (BatchNormalization(name= 'batch_normalization_2'))(c3)
    c3 = (Activation('relu',name = 'activation_2'))(c3)
    c4 = (Conv2D(128, 3, padding="same",name = 'conv2d_3'))(c3)
    c4 = (BatchNormalization(name= 'batch_normalization_3'))(c4)
    c4 = (Activation('relu',name = 'activation_3'))(c4)
    m2 = (MaxPooling2D(pool_size=2, name = 'max_pooling2d_1'))(c4)
    d2 = (Dropout(0.3, name = 'dropout_1'))(m2)
    f1 = GlobalAveragePooling2D()(d2)
    res1 = (Dense(50, activation='relu',name = 'dense_2'))(f1)
    res1 = (Dropout(0.3, name = 'dropout_4'))(res1)
    res1 = (Dense(2, activation='softmax', name = 'coarse'))(res1)

    model = Model(input_img, [res1])
    return(model)



def middle_modelBottleneck( input_shape=(32, 32, 3)):
    input_img = Input(input_shape,name = 'input_1')
    c1 = (Conv2D(32, kernel_size=3, input_shape=input_shape, padding="same",name = 'conv2d'))(input_img)
    c1 = (BatchNormalization(name= 'batch_normalization' ))(c1)
    c1 = (Activation('relu',name = 'activation'))(c1)
    c2 = (Conv2D(64, 3, padding="same", name = 'conv2d_1'))(c1)
    c2 = (BatchNormalization(name= 'batch_normalization_1'))(c2)
    c2 = (Activation('relu',name = 'activation_1'))(c2)
    m1 = (MaxPooling2D(pool_size=2, name = 'max_pooling2d'))(c2)
    d1 = (Dropout(0.3, name = 'dropout'))(m1)

    c3 = (Conv2D(64, 3,padding="same",name = 'conv2d_2'))(d1)
    c3 = (BatchNormalization(name= 'batch_normalization_2'))(c3)
    c3 = (Activation('relu',name = 'activation_2'))(c3)
    c4 = (Conv2D(128, 3, padding="same",name = 'conv2d_3'))(c3)
    c4 = (BatchNormalization(name= 'batch_normalization_3'))(c4)
    c4 = (Activation('relu',name = 'activation_3'))(c4)
    m2 = (MaxPooling2D(pool_size=2, name = 'max_pooling2d_1'))(c4)
    d2 = (Dropout(0.3, name = 'dropout_1'))(m2)

    c5 = (Conv2D(128, 3,padding="same",name = 'conv2d_4'))(d2)
    c5 = (BatchNormalization(name= 'batch_normalization_4'))(c5)
    c5 = (Activation('relu',name = 'activation_4'))(c5)
    c6 = (Conv2D(256, 3, padding="same",name = 'conv2d_5'))(c5)
    c6 = (BatchNormalization(name= 'batch_normalization_5'))(c6)
    c6 = (Activation('relu',name = 'activation_5'))(c6)
    m3 = (MaxPooling2D(pool_size=2, name = 'max_pooling2d_2'))(c6)
    d3 = (Dropout(0.3, name = 'dropout_2'))(m3)

    f1 = GlobalAveragePooling2D()(d2)
    res1 = (Dense(50, activation='relu',name = 'dense_2'))(f1)
    res1 = (Dropout(0.3, name = 'dropout_4'))(res1)
    res1 = (Dense(2, activation='softmax', name = 'coarse'))(res1)
    f2 = GlobalAveragePooling2D()(d3)
    res2 = (Dense(50, activation='relu',name = 'dense_1'))(f2)
    res2 = (Dropout(0.3, name = 'dropout_5'))(res2)
    res2 = (Dense(4, activation='softmax', name = 'middle'))(res2)

    model = Model(input_img, [res1,res2])
    return(model)

def fine_modelBottleneck(input_shape=(32, 32, 3)):
    input_img = Input(input_shape,name = 'input_1')
    c1 = (Conv2D(32, kernel_size=3, input_shape=input_shape, padding="same",name = 'conv2d'))(input_img)
    c1 = (BatchNormalization(name= 'batch_normalization' ))(c1)
    c1 = (Activation('relu',name = 'activation'))(c1)
    c2 = (Conv2D(64, 3, padding="same", name = 'conv2d_1'))(c1)
    c2 = (BatchNormalization(name= 'batch_normalization_1'))(c2)
    c2 = (Activation('relu',name = 'activation_1'))(c2)
    m1 = (MaxPooling2D(pool_size=2, name = 'max_pooling2d'))(c2)
    d1 = (Dropout(0.3, name = 'dropout'))(m1)

    c3 = (Conv2D(64, 3,padding="same",name = 'conv2d_2'))(d1)
    c3 = (BatchNormalization(name= 'batch_normalization_2'))(c3)
    c3 = (Activation('relu',name = 'activation_2'))(c3)
    c4 = (Conv2D(128, 3, padding="same",name = 'conv2d_3'))(c3)
    c4 = (BatchNormalization(name= 'batch_normalization_3'))(c4)
    c4 = (Activation('relu',name = 'activation_3'))(c4)
    m2 = (MaxPooling2D(pool_size=2, name = 'max_pooling2d_1'))(c4)
    d2 = (Dropout(0.3, name = 'dropout_1'))(m2)

    c5 = (Conv2D(128, 3,padding="same",name = 'conv2d_4'))(d2)
    c5 = (BatchNormalization(name= 'batch_normalization_4'))(c5)
    c5 = (Activation('relu',name = 'activation_4'))(c5)
    c6 = (Conv2D(256, 3, padding="same",name = 'conv2d_5'))(c5)
    c6 = (BatchNormalization(name= 'batch_normalization_5'))(c6)
    c6 = (Activation('relu',name = 'activation_5'))(c6)
    m3 = (MaxPooling2D(pool_size=2, name = 'max_pooling2d_2'))(c6)
    d3 = (Dropout(0.3, name = 'dropout_2'))(m3)
    c7 = (Conv2D(256, 3,padding="same",name = 'conv2d_6'))(d3)
    c7 = (BatchNormalization(name= 'batch_normalization_6'))(c7)
    c7 = (Activation('relu',name = 'activation_6'))(c7)

    f1 = GlobalAveragePooling2D()(d2)
    res1 = (Dense(50, activation='relu',name = 'dense_2'))(f1)
    res1 = (Dropout(0.3, name = 'dropout_4'))(res1)
    res1 = (Dense(2, activation='softmax', name = 'coarse'))(res1)
    f2 = GlobalAveragePooling2D()(d3)
    res2 = (Dense(50, activation='relu',name = 'dense_1'))(f2)
    res2 = (Dropout(0.3, name = 'dropout_5'))(res2)
    res2 = (Dense(4, activation='softmax', name = 'middle'))(res2)
    f3 = GlobalAveragePooling2D()(c7)
    res3 = (Dense(100, activation='relu',name = 'dense'))(f3)
    res3 = (Dropout(0.3, name = 'dropout_3'))(res3)
    res3 = (Dense(10, activation='softmax', name = 'fine'))(res3)




    model = Model(input_img, [res1,res2,res3])
    return(model)

def fine_modelBottleneck_single(input_shape=(32, 32, 3)):
    input_img = Input(input_shape,name = 'input_1')
    c1 = (Conv2D(32, kernel_size=3, input_shape=input_shape, padding="same",name = 'conv2d'))(input_img)
    c1 = (BatchNormalization(name= 'batch_normalization' ))(c1)
    c1 = (Activation('relu',name = 'activation'))(c1)
    c2 = (Conv2D(64, 3, padding="same", name = 'conv2d_1'))(c1)
    c2 = (BatchNormalization(name= 'batch_normalization_1'))(c2)
    c2 = (Activation('relu',name = 'activation_1'))(c2)
    m1 = (MaxPooling2D(pool_size=2, name = 'max_pooling2d'))(c2)
    d1 = (Dropout(0.3, name = 'dropout'))(m1)

    c3 = (Conv2D(64, 3,padding="same",name = 'conv2d_2'))(d1)
    c3 = (BatchNormalization(name= 'batch_normalization_2'))(c3)
    c3 = (Activation('relu',name = 'activation_2'))(c3)
    c4 = (Conv2D(128, 3, padding="same",name = 'conv2d_3'))(c3)
    c4 = (BatchNormalization(name= 'batch_normalization_3'))(c4)
    c4 = (Activation('relu',name = 'activation_3'))(c4)
    m2 = (MaxPooling2D(pool_size=2, name = 'max_pooling2d_1'))(c4)
    d2 = (Dropout(0.3, name = 'dropout_1'))(m2)

    c5 = (Conv2D(128, 3,padding="same",name = 'conv2d_4'))(d2)
    c5 = (BatchNormalization(name= 'batch_normalization_4'))(c5)
    c5 = (Activation('relu',name = 'activation_4'))(c5)
    c6 = (Conv2D(256, 3, padding="same",name = 'conv2d_5'))(c5)
    c6 = (BatchNormalization(name= 'batch_normalization_5'))(c6)
    c6 = (Activation('relu',name = 'activation_5'))(c6)
    m3 = (MaxPooling2D(pool_size=2, name = 'max_pooling2d_2'))(c6)
    d3 = (Dropout(0.3, name = 'dropout_2'))(m3)
    c7 = (Conv2D(256, 3,padding="same",name = 'conv2d_6'))(d3)
    c7 = (BatchNormalization(name= 'batch_normalization_6'))(c7)
    c7 = (Activation('relu',name = 'activation_6'))(c7)

    f3 = GlobalAveragePooling2D()(c7)
    res3 = (Dense(100, activation='relu',name = 'dense'))(f3)
    res3 = (Dropout(0.3, name = 'dropout_3'))(res3)
    res3 = (Dense(10, activation='softmax', name = 'fine'))(res3)




    model = Model(input_img, [res1,res2,res3])
    return(model)
