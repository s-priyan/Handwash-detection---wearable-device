# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 16:00:23 2020

@author: Mathanraj-Sharma
"""

import os
import random
import numpy as np
os.environ["TF_KERAS"] = '1'
from datetime import datetime
from keras_radam import RAdam
import tensorflow.keras as keras
from classifiers.BASE import BASE
from tensorflow.keras.layers import Conv1D, Dense, Dropout, Input, GlobalMaxPooling1D, MaxPooling1D, Add
from tensorflow.keras.models import Model

seed_value = 0
random.seed(seed_value)
os.environ['PYTHONHASHSEED'] = str(seed_value)
np.random.seed(seed_value)

class Multi_Headed_CNN(BASE):
    
    def __init__(self, inputs_lens = [104, 104, 104], fsizes = [3,5,7], **kwargs):
        """
        Create an object of Multi_Headed_CNN model

        Parameters
        ----------
        inputs_lens : list/iterable, optional
            Number of data points in each window. The default is [104, 104, 104].
        fsizes : list/iterable, optional
            CNN kernal sizes for each head. The default is [3,5,7].
        **kwargs : dict
            Arguments that are expected in BASE class.

        Returns
        -------
        Object of Multi_Headed_CNN model.

        """
        self.inputs_lens = inputs_lens
        self.fsizes = fsizes
        super().__init__(**kwargs)
        
        
        
    def get_base_model(self, input_len, fsize, n_features):
        """
        Create baseline/residual model for Multiheaded CNN.

        Parameters
        ----------
        input_len : int
            Data points in each window / Timestep.
        fsize : int
            Kernal size to be used in CNN layer.
        n_features : int
            Number of features in data.

        Returns
        -------
        model : keras.models.Model
            Baseline model with specified architecture.

        """
      
        input_seq = Input(shape=(input_len, n_features))

        #choose the number of convolution filters
        nb_filters = 32
        
        #1-D convolution and global max-pooling
        convolved_1 = Conv1D(nb_filters, fsize, padding="same", activation="relu")(input_seq)
        max_pool_1 = MaxPooling1D(pool_size=2)(convolved_1)
        convolved_2 = Conv1D(filters=32, kernel_size=fsize, activation='relu')(max_pool_1)
        max_pool_2 = MaxPooling1D(pool_size=2)(convolved_2)
        convolved_3 = Conv1D(filters=16, kernel_size=fsize, activation='relu')(max_pool_2) 
        processed = GlobalMaxPooling1D()(convolved_3)
        
        #dense layer with dropout regularization
        compressed = Dense(50, activation="relu")(processed)
        compressed = Dropout(0.3)(compressed)
        model = Model(inputs=input_seq, outputs=compressed)
        return model

    
    def build_model(self, input_shape, nb_classes):
        """
        Create an object of Multi_Headed_CNN.

        Parameters
        ----------
        input_shape : list/iterable
            Input shape of train and test data (time_step and no_features). Example [104, 8].
        nb_classes : int
            Number of classes in target/labels.

        Returns
        -------
        model : keras.model.Models
            Multi_Headed_CNN model built using keras.

        """
        
        #the inputs to the branches are the original time series, and its down-sampled versions
        input_smallseq = Input(shape=(self.inputs_lens[0], input_shape[1]))
        input_medseq = Input(shape=(self.inputs_lens[1] , input_shape[1]))
        input_origseq = Input(shape=(self.inputs_lens[2], input_shape[1]))
        
        #the more down-sampled the time series, the shorter the corresponding filter
        base_net_small = self.get_base_model(self.inputs_lens[0], self.fsizes[0], input_shape[1])
        base_net_med = self.get_base_model(self.inputs_lens[1], self.fsizes[1], input_shape[1])
        base_net_original = self.get_base_model(self.inputs_lens[2], self.fsizes[2], input_shape[1])
        embedding_small = base_net_small(input_smallseq)
        embedding_med = base_net_med(input_medseq)
        embedding_original = base_net_original(input_origseq)
        
        #Add all the outputs
        merged = Add()([embedding_small, embedding_med, embedding_original])
        out = Dense(nb_classes, activation='softmax')(merged)
        model = Model(inputs=[input_smallseq, input_medseq, input_origseq], outputs=out)
        
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
        self.data['X_train_shape'] = X_train.shape
        self.data['X_val_shape'] = X_val.shape
        
        start_time = datetime.now()
        
        self.history = self.model.fit(
                [X_train, X_train, X_train],
                y_train,
                batch_size=self.batch_size,
                epochs=self.nb_epochs,
                verbose=self.verbose,
                validation_data=([X_val, X_val, X_val],y_val),
                callbacks=self.callbacks
            )
        
        duration = datetime.now() - start_time
        print(f'Total time taken to train = {duration}')
        self.save_model_params()
        
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
        
        return self.model.predict([X_test, X_test, X_test])
    
    
    def load_weights(self, path):
        return self.model.load_weights(path)
