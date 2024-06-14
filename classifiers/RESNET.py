# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 19:00:23 2020

@author: Mathanraj-Sharma
"""


import random
import numpy as np
import os
os.environ["TF_KERAS"]='1'
from keras_radam import RAdam
import tensorflow.keras as keras
from classifiers.BASE import BASE

seed_value = 0
random.seed(seed_value)
os.environ['PYTHONHASHSEED'] = str(seed_value)
np.random.seed(seed_value)

class RESNET(BASE):
    
             
    def build_model(self, input_shape, nb_classes):
        """
        Create new model object of RESNET.

        Parameters
        ----------
        input_shape : list or tuple
            Input shape of train and test data (time_step and no_features). The default is [104, 8].
        nb_classes : int
            Number of target variables. The default is 11.

        Returns
        -------
        model : keras.models.Model
            RESNET model built using keras.
        """
        
        input_layer = keras.layers.Input(input_shape)
    
        n_filters = 16
        conv1 = keras.layers.Conv1D(filters=n_filters*2, kernel_size=8, padding='same')(input_layer)
        conv1 = keras.layers.BatchNormalization()(conv1)
        conv1 = keras.layers.Activation(activation='relu')(conv1)
    
        conv2 = keras.layers.Conv1D(filters=n_filters*4, kernel_size=5, padding='same')(conv1)
        conv2 = keras.layers.BatchNormalization()(conv2)
        conv2 = keras.layers.Activation('relu')(conv2)
    
        conv3 = keras.layers.Conv1D(filters=n_filters*2, kernel_size=3,padding='same')(conv2)
        conv3 = keras.layers.BatchNormalization()(conv3)
        conv3 = keras.layers.Activation('relu')(conv3)
    
        gap_layer = keras.layers.GlobalAveragePooling1D()(conv3)
    
        output_layer = keras.layers.Dense(nb_classes, activation='softmax')(gap_layer)
        model = keras.models.Model(inputs=input_layer, outputs=output_layer)
        
        opt = RAdam()
        model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
        
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        check_point = keras.callbacks.ModelCheckpoint(self.output_dir+'best_model.hdf5', monitor='val_loss', save_best_only=True)
        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=50, min_lr=0.0001)
        
        self.callbacks = [check_point, reduce_lr]
        
        return model
    
    