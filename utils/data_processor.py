# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 20:29:26 2020

@author: Mathanraj-Sharma
"""

import os
import numpy as np
import json
from sklearn.preprocessing import OneHotEncoder




class data_processor():
    
    def __init__(self, input_dir, output_dir, axis=6):
        """
        Create an instance of data_processor.

        Parameters
        ----------
        input_dir : str
            Path of windowed training and testing data in individual folders.
        output_dir : str
            Path to save model_params (should be same as provvided for model out_dir).
        axis : int, optional
            Number of axis considered for model building. The default is 6.

        Returns
        -------
        data_processor instance.

        """
        
        self.train_data = f'{input_dir}/train/'
        self.test_data = f'{input_dir}/test/'
        self.output_dir = output_dir
        self.axis = axis
        
        self.X_train, self.y_train = self.merge_npz_files(self.train_data, add_zero=False)
        self.X_test, self.y_test = self.merge_npz_files(self.test_data, add_zero=False)
        
        self.one_hot_encoder = OneHotEncoder(sparse=False) 
        
        self.median = None
        self.IQR = None
        self.data = dict()
        
        
    def merge_npz_files(self, path, name=None, add_zero=True):
        """
        Merge windowed data saved in .npz files and create training and testing data. 
        If name argument is provided, filter only the data belongs to that name.

        Parameters
        ----------
        path : str
            Path to get windowed data files.
        name : str, optional
            Name of the person need to be filtered for merging. The default is None.
        add_zero : boolean, optional
            If named person's null action is missing, then pass True to use default null action. The default is True.

        Returns
        -------
        tuple
            Two np.ndarray containing Xs and ys.

        """
        if name:
            files = [path+f.name for f in os.scandir(path) if (f.is_file() and name+'_' in f.name)]
        else:
            files = [path+f.name for f in os.scandir(path) if f.is_file()]
            
        windows, labels = [], []
        zero_flag = False
        
        for file in files:
            if '_0.npz' in file:
                zero_flag = True
                
            sample = np.load(file)
            for window in [*sample['windows']]:
                windows.append(window)
            for label in [*sample['labels']]:
                labels.append(label)
        
        if add_zero:
            if not zero_flag:
                sample = np.load(f'{path}Jey_MALE_24_RIGHT_RIGHT_TIGHT_NO_2020-9-24-20-8-25__0.npz')
                for window in [*sample['windows']]:
                    windows.append(window)
                for label in [*sample['labels']]:
                    labels.append(label)
            
        return np.array(windows), np.array(labels)
     
    def convert_data(self):
        """
        Convert datatypes of X_train, X_test to float32 and y_trian, y_test to int.

        Returns
        -------
        None.

        """
        self.y_train = self.y_train.astype(int)
        self.y_test = self.y_test.astype(int)
        
        self.X_train = np.float32(self.X_train)
        self.X_test = np.float32(self.X_test)
        
       
    def apply_robust_scaling(self):
        """
        Apply Robust Scaling to X_train and X_test.

        Returns
        -------
        None.

        """
        # calculate the median and InterQuartileRange for each axis 
        self.median = np.nanmedian(self.X_train.reshape(-1, self.X_train.shape[-1]), axis=0)
        q1 = np.nanpercentile(self.X_train.reshape(-1, self.X_train.shape[-1]), 25, axis=0)
        q3 = np.nanpercentile(self.X_train.reshape(-1, self.X_train.shape[-1]), 75, axis=0)
        self.IQR = q3 - q1
        
        a = self.axis
        
        # scale the train and testing data using calculated median and IQR
        self.X_train[:,:,0:a] = ((self.X_train[:,:,0:a].reshape(-1, self.X_train[:,:,0:a].shape[-1]) - self.median[:a]) 
                                 / self.IQR[:a]).reshape(self.X_train[:,:,0:a].shape)
        self.X_test[:,:,0:a] = ((self.X_test[:,:,0:a].reshape(-1, self.X_test[:,:,0:a].shape[-1]) - self.median[:a]) 
                                 / self.IQR[:a]).reshape(self.X_test[:,:,0:a].shape)
        
    def get_class_distribution(self):
        """
        Calculate data points for each class (action) and store it in data dictionary. 

        Returns
        -------
        None.

        """
        labels, counts = np.unique(self.y_train, return_counts=True)
        self.data['y_train_distibution'] = dict(zip(labels.tolist(), counts.tolist()))
        
        labels, counts = np.unique(self.y_test, return_counts=True)
        self.data['y_test_distibution'] = dict(zip(labels.tolist(), counts.tolist()))
        
    def one_hot_encode(self):
        """
        Apply OneHotEncoding for y_trian and y_test.

        Returns
        -------
        None.

        """
        # apply one hot encoding for training and testing data
        self.one_hot_encoder = self.one_hot_encoder.fit(self.y_train)
        self.y_train = self.one_hot_encoder.transform(self.y_train)
        self.y_test = self.one_hot_encoder.transform(self.y_test)
        
    def fit_transform(self):
        """
        Apply all preprocessing steps in order. 

        Returns
        -------
        None.

        """
        self.convert_data()
        self.get_class_distribution()
        self.apply_robust_scaling()
        self.one_hot_encode()
        self.save_model_params()
        
   
        
    def prepare_for_inference(self, path, name, add_zero=True, encode_labels=True):
        """
        Load testing data and apply preprocessing specified.

        Parameters
        ----------
        path : str
            Path to read windowed .npz files.
        name : str, optional
            Name of the person need to be filtered for merging. The default is None.
        add_zero : boolean, optional
            If named person's null action is missing, then pass True to use default null action. The default is True.
        encode_labels : boolean, optional
            Apply or not to apply OneHotEncoding to y (labels). The default is True.

        Returns
        -------
        Xp_test : np.ndarray
            Testing features/Windows.
        yp_test : nd.ndarray
            Labels for testing data.

        """
        Xp_test, yp_test = self.merge_npz_files(path, name, add_zero)
        yp_test = yp_test.astype(int)
        
        if encode_labels:
            yp_test = self.one_hot_encoder.transform(yp_test)
        
        Xp_test = np.float32(Xp_test)
        a = self.axis
        Xp_test[:,:,0:a] = ((Xp_test[:,:,0:a].reshape(-1, Xp_test[:,:,0:a].shape[-1]) - self.median[:a]) / self.IQR[:a]).reshape(Xp_test[:,:,0:a].shape)
        
        
        return Xp_test, yp_test
    
    
    def save_model_params(self):
        """
        Save recorded informations regarding data preprocessing to model_params.json file. 
        Median, IQR, train_distribution and test_distribution

        Returns
        -------
        None.

        """
        self.data['median'] = self.median.tolist()[:-2]
        self.data['IQR'] = self.IQR.tolist()[:-2]
        
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        with open(self.output_dir+'model_params.json', 'w') as file:
            json.dump(self.data, file, indent=4, sort_keys=True)
        