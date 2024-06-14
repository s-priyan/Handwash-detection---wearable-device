# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 19:00:23 2020

@author: Mathanraj-Sharma
"""

import os
import random
import numpy as np
import keras_radam
#os.environ["TF_KERAS"] = '1'
from keras_radam import RAdam
import tensorflow.keras as keras
from classifiers.BASE import BASE

seed_value = 0
random.seed(seed_value)
os.environ['PYTHONHASHSEED'] = str(seed_value)
np.random.seed(seed_value)

class FCN(BASE):
    
    def __init__(self, filter_list=[32,32,32], **kwargs):
        """
        Create an object of FCN model

        Parameters
        ----------
        filter_list : list of ints, optional
            Number of filters that need to be used in 1DCNN layers of FCN. The default is [32,32,32].
        **kwargs : dict
            Arguments that are expected in BASE class.

        Returns
        -------
        Object of FCN model.

        """
        self.filter_list = filter_list
        super().__init__(**kwargs)
        
    
    def build_model(self, input_shape, nb_classes):
        """
        Create new model object of FCN.

        Parameters
        ----------
        input_shape : list or tuple
            Input shape of train and test data (time_step and no_features). The default is [104, 8].
        nb_classes : int
            Number of target variables. The default is 11.

        Returns
        -------
        model : keras.models.Model
            FCN model built using keras.

        """
        
        input_layer = keras.layers.Input(input_shape)
    
        conv1 = keras.layers.Conv1D(filters=self.filter_list[0], kernel_size=8, padding='same')(input_layer)
        conv1 = keras.layers.BatchNormalization()(conv1)
        conv1 = keras.layers.Activation(activation='relu')(conv1)
    
        conv2 = keras.layers.Conv1D(filters=self.filter_list[1], kernel_size=5, padding='same')(conv1)
        conv2 = keras.layers.BatchNormalization()(conv2)
        conv2 = keras.layers.Activation('relu')(conv2)
    
        conv3 = keras.layers.Conv1D(filters=self.filter_list[2], kernel_size=3,padding='same')(conv2)
        conv3 = keras.layers.BatchNormalization()(conv3)
        conv3 = keras.layers.Activation('relu')(conv3)
    
        gap_layer = keras.layers.GlobalAveragePooling1D()(conv3)
    
        output_layer = keras.layers.Dense(nb_classes, activation='softmax')(gap_layer)
        model = keras.models.Model(inputs=input_layer, outputs=output_layer)
        
        opt = RAdam()
        model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
        
        # create output folder if not exists
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        check_point = keras.callbacks.ModelCheckpoint(self.output_dir+'best_model.hdf5', monitor='val_loss', save_best_only=True)
        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=50, min_lr=0.0001)
        
        self.callbacks = [check_point, reduce_lr]
        
        return model
   
