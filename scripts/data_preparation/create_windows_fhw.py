# -*- coding: utf-8 -*-
"""
Created on Sat Oct 31 20:59:44 2020

@author: Mathanraj-Sharma
"""
import os
import pandas as pd
import numpy as np
from scipy import stats
import argparse

def file_paths(read_from, actions, version='old'):
    """
    Filter only the files needed

    Parameters
    ----------
    read_from : str
        path of the the dataset.
    actions : list
        action types needed.

    Returns
    -------
    list of filepaths corresponding to the actions.
    """
    file_names = [f.name for f in os.scandir(read_from) if f.is_file()]
    
    if version == 'old':
        filtered_paths = [read_from+name for action in actions 
                      for name in file_names if '_'+action+'_' in name]
        
    else:
        filtered_paths = [read_from+name for action in actions 
                      for name in file_names if ('_'+str(action)+'.csv' in name)
                      or ('-'+str(action)+'.csv' in name)]
        
    return filtered_paths

def create_windows(X, y, time_steps=128, step=64):
    """
    create windowed data

    Parameters
    ----------
    X : pd.Series
        Series of features.
    y : str
        label.
    time_steps : int, optional
        window size. The default is 128.
    step : int, optional
        points to move on each step. The default is 64.

    Returns
    -------
    tuple
        tuple of np.array containing windows and corresponding labels

    """
    Xs, ys = [], []

    for i in range(0, len(X) - time_steps, step):
        v = X.iloc[i:(i + time_steps)].values
        Xs.append(v)
        labels = y.iloc[i: i + time_steps]
        ys.append(stats.mode(labels)[0][0])
        # ys.append(stats.mode(y.iloc[i:(i + time_steps)][0][0]))
    
    return np.array(Xs), np.array(ys).reshape(-1, 1)

def create_windows_combined(X, y, time_steps=128, step=64):
    """
    create windowed data, combines pattern 5A & 5B together

    Parameters
    ----------
    X : pd.Series
        Series of features.
    y : str
        label.
    time_steps : int, optional
        window size. The default is 128.
    step : int, optional
        points to move on each step. The default is 64.

    Returns
    -------
    tuple
        tuple of np.array containing windows and corresponding labels

    """
    Xs, ys = [], []
    
    for i in range(0, len(X) - time_steps, step):
        v = X.iloc[i:(i + time_steps)].values
        Xs.append(v)
        label = stats.mode(y.iloc[i:(i + time_steps)].values)
        
        if label == '7' or label == '8':
            ys.append(7)
        else:
            ys.append(label)
    
    return np.array(Xs), np.array(ys).reshape(-1, 1)

if __name__ == "__main__":
    

    read_from = "F:/intern/senzanalytics-handwash_recognition-cc375ddb4bbc/senzanalytics-handwash_recognition-cc375ddb4bbc/data/data-collection-23-09-2020/no_duplicates/"
    write_to = "F:/intern/senzanalytics-handwash_recognition-cc375ddb4bbc/senzanalytics-handwash_recognition-cc375ddb4bbc/data/data-collection-23-09-2020/windowed_data/"
    action_str = "0,1,2,3,4,5,6,7,8,9,10"
    version = "new"
    axis = '6'
    combine  = '0'
    hand_info = '1'
    
    # list of actions needed
    actions = action_str.split(',')

    print('Starting to create windows...')
    if not os.path.exists(write_to):
            os.makedirs(write_to)
            
    if version == 'new':
        if axis == '6':
            cols = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']
            print(f'axis = {axis}')
        else:
            cols = ['acc_x', 'acc_y', 'acc_z']
    else:
        cols = ['X (mg)', 'Y (mg)', 'Z (mg)']
    
    for action in actions:
        filtered_paths = file_paths(read_from, [action], version)
        
        if version == 'new':
            for file_path in filtered_paths:
                file_basename = os.path.basename(file_path)
                dominant_hand = file_basename.split('_')[3]
                wearing_hand = file_basename.split('_')[4]  
                
                if ('--'+action+'.csv') or ('__'+action+'.csv') or ('_'+action+'.csv') in file_basename:
                    print(f'Processing file {file_basename} for action {action}')
                    df = pd.read_csv(file_path)
                    # df.drop(df.head(52).index, inplace=True)
                    # df.drop(df.tail(52).index, inplace=True)
                    
                    # skip files containing data less than 2sec
                    if df.shape[0] < 100:
                        continue
                    
                    if hand_info == '1':
                        df.loc[:,'d_hand'] = 1 if dominant_hand == 'RIGHT' else 0
                        df.loc[:,'w_hand'] = 1 if wearing_hand == 'RIGHT' else 0
                        cols = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z', 'd_hand', 'w_hand']
                    
                    if combine == '1':
                        windows, labels = create_windows_combined(df[cols], df['label_0'], 104, 104//3)
                    elif combine == '0':
                        windows, labels = create_windows(df[cols], df['label_0'], 104, 104//3)
                    else:
                        raise ValueError ('Invalid entry for argument -m ')
                    
                    np.savez(write_to+file_basename[:-4]+'.npz', windows=windows, labels=labels)
        else:
            for file_path in filtered_paths:
                file_basename = os.path.basename(file_path)
                
                if ('_'+action+'_') in file_basename:
                    print(f'Processing file {file_basename} for action {action}')
                    df = pd.read_csv(file_path, skiprows=4)
                    df.drop(df.head(104).index, inplace=True)
                    df.drop(df.tail(104).index, inplace=True)
                    # df.set_index('count', inplace=True)
                    # df.sort_index(inplace=True)
                    
                    windows, labels = create_windows(df[cols], action, 104, 104//3)
                    np.savez(write_to+file_basename[:-4]+'.npz', windows=windows, labels=labels)
            
                
    
    print(f'Finished and files are saved to {write_to}')
    
    
# =============================================================================
# python create_windows.py -s F:/intern/senzanalytics-handwash_recognition-cc375ddb4bbc/senzanalytics-handwash_recognition-cc375ddb4bbc/data/data-collection-23-09-2020/no_duplicates/ -d F:/intern/senzanalytics-handwash_recognition-cc375ddb4bbc/senzanalytics-handwash_recognition-cc375ddb4bbc/data/data-collection-23-09-2020/windowed_data/ -a "0,1,2,3,4,5,6,7,8,9,10" -v "new" -c 6 -m 0 -x 1
# =============================================================================
