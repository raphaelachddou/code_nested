import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.datasets import mnist
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Input, Concatenate, Lambda, Softmax, Activation
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Cropping2D, GlobalAveragePooling2D
from tensorflow.keras import backend as K
from time import time
from utils import custom_metric

class FashionMnist_classifier_coarse:
    """
    class of the first part of the network only trainable on the first bottleneck
    object methods :
    - init : initialize the model
    - loss : customized loss : made for batch_size = 32 only
    - buildModel : keras implementation of the model with the right trainable parameters
    """
    def __init__(self):
        self.img_cols = 28
        self.img_rows = 28
        self.model = self.buildModel()
    def loss(self, y_true, y_pred):
        n = K.shape(y_true)[0]
        n = tf.cast(n,tf.float32)
        cum_loss = tf.constant(0.0)
        # for i in range(32):
        #     loss = tf.losses.softmax_cross_entropy(y_true[i,1:],y_pred[i,1:]) + (y_pred[i,0])**2
        #     cum_loss+= (y_true[i,0]*loss)
        loss = tf.math.reduce_mean(tf.losses.softmax_cross_entropy(y_true[:,1:],y_pred[:,1:]) + (y_pred[:,0])**2)
        return(loss)
    def buildModel(self):
        input_shape = (self.img_rows,self.img_cols,1)
        input_img = Input(shape=input_shape,name = "Input", dtype = 'float32')
        conv1 = Conv2D(4, (3, 3),padding='same', activation='relu',name = "conv2d_1", trainable=True)(input_img)
        conv2 = Conv2D(8, (3, 3),padding='same', activation='relu',name = "conv2d_2", trainable=True)(conv1)
        conv2bis = MaxPooling2D(pool_size=(2, 2),name = "max_pooling2d_1")(conv2)
        conv3 = Conv2D(16, (3, 3),padding='same', activation='relu',name = "conv2d_3", trainable=True)(conv2bis)
        conv3bis = MaxPooling2D(pool_size=(2, 2),name = "max_pooling2d_2")(conv3)
        conv4 = Conv2D(32, (3, 3),padding='same', activation='relu',name = "conv2d_4", trainable=True)(conv3bis)
        conv4bis = MaxPooling2D(pool_size=(2, 2),name = "max_pooling2d_3")(conv4)
        conv5 = Conv2D(64, (3, 3),padding='same', activation='relu',name = "conv2d_5", trainable=True)(conv4bis)
        #res1 = GlobalAveragePooling2D()(conv5)
        res1 = Flatten()(conv5)
        res1 = Dense(50,name = "fc0")(res1)
        res1 = Dense(3,name = "fc1")(res1)
        #res1 = Lambda('softmax',name = 'coarse')(res1)
        res1 = Activation('softmax',name = 'coarse')(res1)
        #res2 = Activation('softmax')(res1)
        conv5bis = UpSampling2D(size=(2, 2), name = 'up_sampling2d_1')(conv5)
        conv4tris = Cropping2D(cropping=((1, 0), (1, 0)))(conv4)
        conv6 = Concatenate(name = 'concatenate_1', axis = 3)([conv5bis,conv4tris])
        conv7 = Conv2D(32, (3, 3),padding='same', activation='relu',name = "conv2d_6",trainable=False)(conv6)
        res2 = Flatten()(conv7)
        #res2 = GlobalAveragePooling2D()(conv7)
        res2 = Dense(30,name = "fc2bis",trainable=False)(res2)
        res2 = Dense(5,name = "fc2",trainable=False)(res2)
        res2 = Activation('softmax',name = '4_class')(res2)
        conv7bis = UpSampling2D(size=(2, 2), name = 'up_sampling2d_2')(conv7)
        conv3tris = Cropping2D(cropping=((1, 1), (1, 1)))(conv3)
        conv8 = Concatenate(name = 'concatenate_2', axis = 3)([conv7bis,conv3tris])
        conv9 = Conv2D(32, (3, 3),padding='same', activation='relu',name = "conv2d_7",trainable=False)(conv8)
        conv10 = Conv2D(16, (3, 3),padding='same', activation='relu',name = "conv2d_8",trainable=False)(conv9)
        res3 = Flatten()(conv9)
        #res3 = GlobalAveragePooling2D()(conv10)
        res3 = Dense(20,name = "fc3bis",trainable=False)(res3)
        res3 = Dense(11,name = "fc3", trainable=False)(res3)
        res3 = Activation('softmax',name = 'fine_grained')(res3)
        final_result = [res1,res2,res3]
        model = Model(inputs=input_img,outputs=final_result)
        return(model)

class FashionMnist_classifier_middle:
    """
    class of the first and second parts of the network only trainable up to the second output
    object methods :
    - init : initialize the model
    - loss : customized loss
    - buildModel : keras implementation of the model with the right trainable parameters
    """
    def __init__(self):
        self.img_cols = 28
        self.img_rows = 28
        self.model = self.buildModel()
    def loss(self, y_true, y_pred):
        n = K.shape(y_true)[0]
        n = tf.cast(n,tf.float32)
        cum_loss = tf.constant(0.0)
        # for i in range(32):
        #     loss = tf.losses.softmax_cross_entropy(y_true[i,1:],y_pred[i,1:]) + (y_pred[i,0])**2
        #     cum_loss+= (y_true[i,0]*loss)
        loss = tf.math.reduce_mean(tf.losses.softmax_cross_entropy(y_true[:,1:],y_pred[:,1:]) + (y_pred[:,0])**2)
        return(loss)

    def buildModel(self):
        input_shape = (self.img_rows,self.img_cols,1)
        input_img = Input(shape=input_shape,name = "Input", dtype = 'float32')
        conv1 = Conv2D(4, (3, 3),padding='same', activation='relu',name = "conv2d_1", trainable=True)(input_img)
        conv2 = Conv2D(8, (3, 3),padding='same', activation='relu',name = "conv2d_2", trainable=True)(conv1)
        conv2bis = MaxPooling2D(pool_size=(2, 2),name = "max_pooling2d_1")(conv2)
        conv3 = Conv2D(16, (3, 3),padding='same', activation='relu',name = "conv2d_3", trainable=True)(conv2bis)
        conv3bis = MaxPooling2D(pool_size=(2, 2),name = "max_pooling2d_2")(conv3)
        conv4 = Conv2D(32, (3, 3),padding='same', activation='relu',name = "conv2d_4", trainable=True)(conv3bis)
        conv4bis = MaxPooling2D(pool_size=(2, 2),name = "max_pooling2d_3")(conv4)
        conv5 = Conv2D(64, (3, 3),padding='same', activation='relu',name = "conv2d_5", trainable=True)(conv4bis)
        #res1 = GlobalAveragePooling2D()(conv5)
        res1 = Flatten()(conv5)
        res1 = Dense(50,name = "fc0")(res1)
        res1 = Dense(3,name = "fc1")(res1)
        #res1 = Lambda('softmax',name = 'coarse')(res1)
        res1 = Activation('softmax',name = 'coarse')(res1)
        #res2 = Activation('softmax')(res1)
        conv5bis = UpSampling2D(size=(2, 2), name = 'up_sampling2d_1')(conv5)
        conv4tris = Cropping2D(cropping=((1, 0), (1, 0)))(conv4)
        conv6 = Concatenate(name = 'concatenate_1', axis = 3)([conv5bis,conv4tris])
        conv7 = Conv2D(32, (3, 3),padding='same', activation='relu',name = "conv2d_6",trainable=True)(conv6)
        res2 = Flatten()(conv7)
        #res2 = GlobalAveragePooling2D()(conv7)
        res2 = Dense(30,name = "fc2bis",trainable=True)(res2)
        res2 = Dense(5,name = "fc2",trainable=True)(res2)
        res2 = Activation('softmax',name = '4_class')(res2)
        conv7bis = UpSampling2D(size=(2, 2), name = 'up_sampling2d_2')(conv7)
        conv3tris = Cropping2D(cropping=((1, 1), (1, 1)))(conv3)
        conv8 = Concatenate(name = 'concatenate_2', axis = 3)([conv7bis,conv3tris])
        conv9 = Conv2D(32, (3, 3),padding='same', activation='relu',name = "conv2d_7",trainable=False)(conv8)
        conv10 = Conv2D(16, (3, 3),padding='same', activation='relu',name = "conv2d_8",trainable=False)(conv9)
        res3 = Flatten()(conv9)
        #res3 = GlobalAveragePooling2D()(conv10)
        res3 = Dense(20,name = "fc3bis",trainable=False)(res3)
        res3 = Dense(11,name = "fc3", trainable=False)(res3)
        res3 = Activation('softmax',name = 'fine_grained')(res3)
        final_result = [res1,res2,res3]
        model = Model(inputs=input_img,outputs=final_result)
        return(model)

class FashionMnist_classifier_fine:

    """
    class of the first and second parts of the network only trainable up to the second output
    object methods :
    - init : initialize the model
    - loss : customized loss
    - buildModel : keras implementation of the model with the right trainable parameters
    """
    def __init__(self):
        self.img_cols = 28
        self.img_rows = 28
        self.model = self.buildModel()
    def loss(self, y_true, y_pred):
        n = K.shape(y_true)[0]
        n = tf.cast(n,tf.float32)
        cum_loss = tf.constant(0.0)
        # for i in range(32):
        #     loss = tf.losses.softmax_cross_entropy(y_true[i,1:],y_pred[i,1:]) + (y_pred[i,0])**2
        #     cum_loss+= (y_true[i,0]*loss)
        loss = tf.math.reduce_mean(tf.losses.softmax_cross_entropy(y_true[:,1:],y_pred[:,1:]) + (y_pred[:,0])**2)
        return(loss)

    def buildModel(self):
        input_shape = (self.img_rows,self.img_cols,1)
        input_img = Input(shape=input_shape,name = "Input", dtype = 'float32')
        conv1 = Conv2D(4, (3, 3),padding='same', activation='relu',name = "conv2d_1", trainable=True)(input_img)
        conv2 = Conv2D(8, (3, 3),padding='same', activation='relu',name = "conv2d_2", trainable=True)(conv1)
        conv2bis = MaxPooling2D(pool_size=(2, 2),name = "max_pooling2d_1")(conv2)
        conv3 = Conv2D(16, (3, 3),padding='same', activation='relu',name = "conv2d_3", trainable=True)(conv2bis)
        conv3bis = MaxPooling2D(pool_size=(2, 2),name = "max_pooling2d_2")(conv3)
        conv4 = Conv2D(32, (3, 3),padding='same', activation='relu',name = "conv2d_4", trainable=True)(conv3bis)
        conv4bis = MaxPooling2D(pool_size=(2, 2),name = "max_pooling2d_3")(conv4)
        conv5 = Conv2D(64, (3, 3),padding='same', activation='relu',name = "conv2d_5", trainable=True)(conv4bis)
        #res1 = GlobalAveragePooling2D()(conv5)
        f1 = Flatten()(conv5)
        d1 = Dense(50,name = "fc0")(f1)
        res1 = Dense(3,name = "fc1")(d1)
        #res1 = Lambda('softmax',name = 'coarse')(res1)
        res1 = Activation('softmax',name = 'coarse')(res1)
        #res2 = Activation('softmax')(res1)
        conv5bis = UpSampling2D(size=(2, 2), name = 'up_sampling2d_1')(conv5)
        conv4tris = Cropping2D(cropping=((1, 0), (1, 0)))(conv4)
        conv6 = Concatenate(name = 'concatenate_1', axis = 3)([conv5bis,conv4tris])
        conv7 = Conv2D(32, (3, 3),padding='same', activation='relu',name = "conv2d_6",trainable=False)(conv6)
        f2 = Flatten()(conv7)
        #res2 = GlobalAveragePooling2D()(conv7)
        d2 = Dense(30,name = "fc2bis",trainable=True)(f2)
        res2 = Dense(5,name = "fc2",trainable=True)(d2)
        res2 = Activation('softmax',name = '4_class')(res2)
        conv7bis = UpSampling2D(size=(2, 2), name = 'up_sampling2d_2')(conv7)
        conv3tris = Cropping2D(cropping=((1, 1), (1, 1)))(conv3)
        conv8 = Concatenate(name = 'concatenate_2', axis = 3)([conv7bis,conv3tris])
        conv9 = Conv2D(32, (3, 3),padding='same', activation='relu',name = "conv2d_7",trainable=True)(conv8)
        conv10 = Conv2D(16, (3, 3),padding='same', activation='relu',name = "conv2d_8",trainable=True)(conv9)
        f3 = Flatten()(conv9)
        #res3 = GlobalAveragePooling2D()(conv10)
        d3 = Dense(20,name = "fc3bis",trainable=True)(f3)
        res3 = Dense(11,name = "fc3", trainable=True)(d3)
        res3 = Activation('softmax',name = 'fine_grained')(res3)
        final_result = [res1,res2,res3,d1,d2,d3]
        model = Model(inputs=input_img,outputs=final_result)
        return(model)



class FashionMnist_classifier_full:
    def __init__(self):
        self.img_cols = 28
        self.img_rows = 28
        self.model = self.buildModel()

    def buildModel(self):
        input_shape = (self.img_rows,self.img_cols,1)
        input_img = Input(shape=input_shape,name = "Input", dtype = 'float32')
        conv1 = Conv2D(4, (3, 3),padding='same', activation='relu',name = "conv2d_1", trainable=True)(input_img)
        conv2 = Conv2D(8, (3, 3),padding='same', activation='relu',name = "conv2d_2", trainable=True)(conv1)
        conv2bis = MaxPooling2D(pool_size=(2, 2),name = "max_pooling2d_1")(conv2)
        conv3 = Conv2D(16, (3, 3),padding='same', activation='relu',name = "conv2d_3", trainable=True)(conv2bis)
        conv3bis = MaxPooling2D(pool_size=(2, 2),name = "max_pooling2d_2")(conv3)
        conv4 = Conv2D(32, (3, 3),padding='same', activation='relu',name = "conv2d_4", trainable=True)(conv3bis)
        conv4bis = MaxPooling2D(pool_size=(2, 2),name = "max_pooling2d_3")(conv4)
        conv5 = Conv2D(64, (3, 3),padding='same', activation='relu',name = "conv2d_5", trainable=True)(conv4bis)
        conv5bis = UpSampling2D(size=(2, 2), name = 'up_sampling2d_1')(conv5)
        conv4tris = Cropping2D(cropping=((1, 0), (1, 0)))(conv4)
        conv6 = Concatenate(name = 'concatenate_1', axis = 3)([conv5bis,conv4tris])
        conv7 = Conv2D(32, (3, 3),padding='same', activation='relu',name = "conv2d_6")(conv6)
        conv7bis = UpSampling2D(size=(2, 2), name = 'up_sampling2d_2')(conv7)
        conv3tris = Cropping2D(cropping=((1, 1), (1, 1)))(conv3)
        conv8 = Concatenate(name = 'concatenate_2', axis = 3)([conv7bis,conv3tris])
        conv9 = Conv2D(32, (3, 3),padding='same', activation='relu',name = "conv2d_7",trainable=True)(conv8)
        conv10 = Conv2D(16, (3, 3),padding='same', activation='relu',name = "conv2d_8",trainable=True)(conv9)
        #res3 = GlobalAveragePooling2D()(conv10)
        res3 = Flatten()(conv10)
        res3 = Dense(20,name = "fc3bis",trainable=True)(res3)
        res3 = Dense(10,name = "fc3", trainable=True)(res3)
        res3 = Activation('softmax',name = 'fine_grained')(res3)
        final_result = res3
        model = Model(inputs=input_img,outputs=final_result)
        return(model)
