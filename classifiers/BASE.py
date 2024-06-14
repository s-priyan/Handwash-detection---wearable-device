# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 19:00:23 2020

@author: Mathanraj-Sharma
"""

import os
import json
import random
from contextlib import redirect_stdout
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
os.environ["TF_KERAS"] = '1'
import tensorflow.keras as keras

seed_value = 0
random.seed(seed_value)
os.environ['PYTHONHASHSEED'] = str(seed_value)
np.random.seed(seed_value)

class BASE(object):
    
    def __init__(self, output_dir, input_shape = [104, 8], nb_classes = 11, axis=6, build=True, batch_size=16, epochs=25, verbose=1):
        """
        Create object of Base Model.

        Parameters
        ----------
        output_dir : str
            Directory where trained model have to be saved.
        input_shape : list or tuple, optional
            Input shape of train and test data (time_step and no_features). The default is [104, 8].
        nb_classes : int, optional
            Number of target variables. The default is 11.
        axis : int, optional
            Number of axises used in training (if only accelerometer 3, if accelerometer and gyrometer then 6). The default is 6.
        build : boolean, optional
            True if build new model for training or False for only inferencing. The default is True.
        batch_size : int, optional
            Number of batch size of training. The default is 16.
        epochs : int, optional
            Number of iterations to train model on training data. The default is 25.
        verbose : int, optional
            1 for log info, 0 for silent while training. The default is 1.

        Returns
        -------
        Object of Base model.

        """
    
        self.output_dir = output_dir
        self.axis = axis
        self.batch_size = batch_size
        self.nb_epochs = epochs
        self.verbose = verbose
        self.data = dict()
        self.callbacks = None
        self.history = None
        
        # create model object if build is True
        if build:
            self.model = self.build_model(input_shape, nb_classes)
           
        
        
        
    def build_model(self, input_shape, nb_classes):
        """
        Create new model object when build is True. Have to be implemented in inherited child class.

        Parameters
        ----------
        input_shape : list or tuple
            Input shape of training and testing data (time_step, no_features).
        nb_classes : int
            Number of target variables.

        Returns
        -------
        None.

        """
        print ("Yet not implemented...")
        
        return None
    
    def save_model_params(self):
        """
        Append model parameters in model_params.json file in output_directory 

        Returns
        -------
        None.

        """
        self.data['output_dir'] = self.output_dir
        self.data['axis'] = self.axis
        self.data['batch_size'] = self.batch_size
        self.data['epochs'] = self.nb_epochs
        
        # read model_params.json created by data_processor 
        with open(self.output_dir+'model_params.json', 'r') as file:
            data = json.load(file)
        
        # merge data from data_processor and this model to a single dictonary    
        final = {**data, **self.data}
        
        # overwrite model_params.json with all merged meta data
        with open(self.output_dir+'model_params.json', 'w') as file:
            json.dump(final, file, indent=4)
    
    
    def save_model_summary(self):
        """
        Save model summary to model_summary.txt in output_directory

        Returns
        -------
        None.

        """
        with open(self.output_dir+'model_summary.txt', 'w') as file:
            with redirect_stdout(file):
                self.model.summary()
        
    
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
        
        start_time = datetime.now()
        
        # traini model
        self.history = self.model.fit(
                X_train,
                y_train,
                batch_size=self.batch_size,
                epochs=self.nb_epochs,
               # verbose=self.verbose,
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
        Predict labels for testing data

        Parameters
        ----------
        X_test : np.array
            Testing data, in passed input shape.

        Returns
        -------
        np.array
            prediced out puts depending on the model specified in build_model. Normally soft_max outputs.

        """
        
        # model = keras.models.load_model(self.output_dir+'best_model.hdf5', compile=False)
        
        return self.model.predict(X_test)
           
            
            
    def plot_graph(self, train, val, title):
        """
        Plot learning curves. (accuracy or loss)

        Parameters
        ----------
        train : list or any iterable
            Record of accuracy or loss.
        val : list or any iterable
            Record of accuracy or loss.
        title : str
            Name of the graph.

        Returns
        -------
        fig : plt.fig
            figure of plot.

        """
        fig = plt.figure(figsize=(7,5))
        _ = plt.plot(train)
        _ = plt.plot(val)
        _ = plt.title(f'Model {title}')
        _ = plt.ylabel(f'{title}')
        _ = plt.xlabel('Epoch')
        _ = plt.legend(['Train', 'Val'], loc='upper right')
        
        return fig
    
        
