# -*- coding: utf-8 -*-
"""
Created on Sun Nov 29 16:11:34 2020

@author: Mathanraj-Sharma
@email: rvmmathanraj@gmail.com
"""

import os
import random
import numpy as np
import tensorflow.keras as keras
from classifiers.BASE import BASE

seed_value = 0
random.seed(seed_value)
os.environ['PYTHONHASHSEED'] = str(seed_value)
np.random.seed(seed_value)

class Bi_LSTM(BASE):
         
    def build_model(self, input_shape, nb_classes):
        """
        Create an object of Bi_LSTM model.

        Parameters
        ----------
        input_shape : list/iterable
             Input shape of train and test data (time_step and no_features). The default is [104, 8].
        nb_classes : int
            Number of target variables.

        Returns
        -------
        Object of Bi_LSTM model.

        """
        model = keras.Sequential()
        model.add(
            keras.layers.Bidirectional(
              keras.layers.LSTM(
                  units=128,
                  input_shape=input_shape
              )
            )
        )
        model.add(keras.layers.Dropout(rate=0.5))
        model.add(keras.layers.Dense(units=128, activation='relu'))
        model.add(keras.layers.Dense(nb_classes, activation='softmax'))
        
        
        model.compile(loss='categorical_crossentropy', optimizer="Adam", metrics=['accuracy'])
        
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        check_point = keras.callbacks.ModelCheckpoint(self.output_dir+'best_model.hdf5', monitor='val_loss', save_best_only=True)
        
        self.callbacks = [check_point]
        
        return model
   