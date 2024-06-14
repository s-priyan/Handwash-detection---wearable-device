# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 15:00:23 2020

@author: Mathanraj-Sharma
"""

import os
import random
import numpy as np
os.environ["TF_KERAS"] = '1'
from keras_radam import RAdam
import tensorflow.keras as keras
from classifiers.BASE import BASE

seed_value = 0
random.seed(seed_value)
os.environ['PYTHONHASHSEED'] = str(seed_value)
np.random.seed(seed_value)

class Simple_CNN(BASE):
        
        
    def build_model(self, input_shape, nb_classes):
        """
        Create an object of Simple_CNN model.

        Parameters
        ----------
        input_shape : list/iterable
             Input shape of train and test data (time_step and no_features). The default is [104, 8].
        nb_classes : int
            Number of target variables.

        Returns
        -------
        Object of Simple_CNN model.

        """
       
        model = keras.models.Sequential()
        model.add(keras.layers.Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=input_shape))
        model.add(keras.layers.MaxPooling1D(pool_size=2))
        model.add(keras.layers.Conv1D(filters=32, kernel_size=3, activation='relu'))
        model.add(keras.layers.MaxPooling1D(pool_size=2))
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dropout(0.5))
        model.add(keras.layers.Dense(100, activation='relu'))
        model.add(keras.layers.Dense(nb_classes, activation='softmax'))
        
        initial_learning_rate = 0.001
        
        lr_schedule = keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate,
            decay_steps=100,
            decay_rate=0.96,
            staircase=True
        )
        opt = RAdam(learning_rate=lr_schedule)
        
        model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
        
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        check_point = keras.callbacks.ModelCheckpoint(self.output_dir+'best_model.hdf5', monitor='val_loss', save_best_only=True)
        
        self.callbacks = [check_point]
        
        return model
   