# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 18:08:23 2020

@author: Mathanraj-Sharma
"""

import os
import random
import numpy as np
from datetime import datetime
os.environ["TF_KERAS"] = '1'
import tensorflow.keras as keras
from classifiers.BASE import BASE

seed_value = 0
random.seed(seed_value)
os.environ['PYTHONHASHSEED'] = str(seed_value)
np.random.seed(seed_value)

class CNN_LSTM(BASE):
    
        
    def build_model(self, input_shape, nb_classes):
        """
        Create an object of CNN_LSTM model.

        Parameters
        ----------
        input_shape : list/iterable
             Input shape of train and test data (time_step and no_features). The default is [104, 8].
        nb_classes : int
            Number of target variables.

        Returns
        -------
        Object of CNN_LSTM model.

        """
        model = keras.models.Sequential()
        
        # define CNN model
        model.add(keras.layers.TimeDistributed(keras.layers.Conv1D(filters=32, kernel_size=3,
                       activation='relu', input_shape=input_shape)))
        model.add(keras.layers.TimeDistributed(keras.layers.BatchNormalization()))
        model.add(keras.layers.TimeDistributed(keras.layers.MaxPooling1D(2)))
        
        
        model.add(keras.layers.TimeDistributed(keras.layers.Conv1D(filters=64, kernel_size=3,
                       activation='relu')))
        model.add(keras.layers.TimeDistributed(keras.layers.BatchNormalization()))
        model.add(keras.layers.TimeDistributed(keras.layers.MaxPooling1D(2)))
        # model.add(keras.layers.Dropout(0.5))
        model.add(keras.layers.TimeDistributed(keras.layers.Flatten()))
        
        # define LSTM model
        model.add(keras.layers.LSTM(units=16))
        model.add(keras.layers.Dropout(0.5))
        model.add(keras.layers.Dense(100))
        model.add(keras.layers.Dense(nb_classes, activation='softmax'))   
        
        opt = 'Adam' #RAdam()
        
        model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
        
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        check_point = keras.callbacks.ModelCheckpoint(self.output_dir+'best_model.hdf5', monitor='val_loss', save_best_only=True)
        
        self.callbacks = [check_point]
        
        return model
   
    
    def fit(self, X_train, y_train, X_val, y_val):
        """
        Fit(train) Model, then call save_model_params and plot_graphs.

        Parameters
        ----------
        X_train : np.array 
            Training data, in passed input shape [Samples, input_shape[0], input_shape[1]].
        y_train : np.array
            OneHotEncoded target variables.
        X_val : np.array
            Validation data, in passed input shape.
        y_val : np.array
            OneHotEncoded target variables.

        Returns
        -------
        None.

        """
        #  add shape information of training and testing data data dictionary
        self.data['X_train_shape'] = X_train.shape
        self.data['X_val_shape'] = X_val.shape
        
        n_steps, n_length = 2, 52
        X_train = X_train.reshape(X_train.shape[0], n_steps, n_length, X_train.shape[2])
        X_val = X_val.reshape(X_val.shape[0], n_steps, n_length, X_val.shape[2])
        
        start_time = datetime.now()
        
        # traini model
        self.history = self.model.fit(
                X_train,
                y_train,
                batch_size=self.batch_size,
                epochs=self.nb_epochs,
                verbose=self.verbose,
                validation_data=(X_val,y_val),
                callbacks=self.callbacks
            )
        
        duration = datetime.now() - start_time
        print(f'Total time taken to train = {duration}')
        
        # save model parameters to model_params.json
        self.save_model_params()
        
        # save model summary
        self.save_model_summary()
        
        # plot and save accuracy and loss curves of training
        fig_loss = self.plot_graph(self.history.history['loss'], self.history.history['val_loss'], title='loss curve')    
        fig_acc = self.plot_graph(self.history.history['accuracy'], self.history.history['val_accuracy'], title='accuracy curve')
        
        fig_loss.savefig(self.output_dir +'loss.png')
        fig_acc.savefig(self.output_dir +'accuracy.png')
        
        keras.backend.clear_session()
        
        
    def predict(self, X_test):
        """
        Predict labels for testing data.

        Parameters
        ----------
        X_test : np.array
            Testing data, in passed input shape.

        Returns
        -------
        np.array
            prediced out puts depending on the model specified in build_model. Normally soft_max outputs.

        """
        n_steps, n_length = 2, 52
        X_test = X_test.reshape(X_test.shape[0], n_steps, n_length, X_test.shape[2])
        # model = keras.models.load_model(self.output_dir+'best_model.hdf5', compile=False)
        
        return self.model.predict(X_test)