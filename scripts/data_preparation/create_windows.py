# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 17:22:19 2020

@author: Mathanraj-Sharma
"""

import os
import pandas as pd
import numpy as np
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
    label = y
    for i in range(0, len(X) - time_steps, step):
        v = X.iloc[i:(i + time_steps)].values
        Xs.append(v)
        ys.append(label)
    
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
    label = y
    for i in range(0, len(X) - time_steps, step):
        v = X.iloc[i:(i + time_steps)].values
        Xs.append(v)
        
        if label == '7' or label == '8':
            ys.append(7)
        else:
            ys.append(label)
    
    return np.array(Xs), np.array(ys).reshape(-1, 1)

if __name__ == "__main__":
    
    ap = argparse.ArgumentParser()
    ap.add_argument(
        '-s', 
        required=True, 
        help='Path for source dataset directory'
    )
    ap.add_argument(
        '-d', 
        required=True, 
        help='File path(including file name) to save aggregated datafile'
    )
    ap.add_argument(
        '-a', 
        required=True, 
        help='List of actions needed Eg 4a, null,1'
    )
    
    ap.add_argument(
        '-v', 
        required=True, 
        help='version of dataset'
    )

    ap.add_argument(
        '-c', 
        required=True, 
        help='cols or no of axis'
    )
    
    ap.add_argument(
        '-m', 
        required=True, 
        help='combine 5A and 5B'
    )
    
    ap.add_argument(
        '-x', 
        required=True, 
        help='add hand information'
    )
    
    args = vars(ap.parse_args())
    read_from = args['s']
    write_to = args['d']
    action_str = args['a']
    version = args['v']
    axis = args['c']
    combine  = args['m']
    hand_info = args['x']
    
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
                    df.drop(df.head(52).index, inplace=True)
                    df.drop(df.tail(52).index, inplace=True)
                    
                    # skip files containing data less than 2sec
                    if df.shape[0] < 100:
                        continue
                    
                    if hand_info == '1':
                        df.loc[:,'d_hand'] = 1 if dominant_hand == 'RIGHT' else 0
                        df.loc[:,'w_hand'] = 1 if wearing_hand == 'RIGHT' else 0
                        if axis == '6':
                            cols = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z', 'd_hand', 'w_hand']
                        else:
                            cols = ['acc_x', 'acc_y', 'acc_z', 'd_hand', 'w_hand']
                    
                    if combine == '1':
                        windows, labels = create_windows_combined(df[cols], action, 104, 104//3)
                    elif combine == '0':
                        windows, labels = create_windows(df[cols], action, 104, 104//3)
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
# python create_windows.py -s ..\..\..\data\10-class\smart_watch_data_sailesh\29-11-2020\ -d ..\..\..\data\10-class\windowed_handinfo_sailesh\train\ -a "0,1,2,3,4,5,6,7,8,9,10" -v "new" -c 6 -m 0 -x 1
# =============================================================================
