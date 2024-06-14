# Handwash Recognition Package

This repository contains the packages, scripts and test notebook used throughout the experiment to build the Deep Learning model to classify the handwash patterns. 

* Author : Shobanapriyan Chandrasegaran.
---

## Setup

We have used a specific version of tensorflow and keras inorder to support with STM CUBE, which we used to compress the model inorder to run in STM Microcontroller of Wrist Band. 
Below are the steps to rebuild our Development environment.

1. Install **Anaconda** package manager
2. Download the [*environment.yml*]
3. Create a virtual environment using conda by importing the above **environment.yml** file.
    * Open Anaconda Terminal in Windows (if ubuntu/linux go to terminal).
    * conda env create -f [path to the environment file] .
    * conda activate tf-gpu.

---

## Quick Start Guide

### Dataset
Use the custom device and software built by team SenzMate for data collection.

1. Collect data for each action individually and save them in csv files using below given naming convention. 
    * **Pattern**: name_gender_age_dominant hand_wearing hand_start date&time_end date&time_label_0.csv.
    * **Example**: *varathepan_male_23_l_l_2020-08-21-10:36:27.59_2020-08-22-10:36:27.59_0.csv*.
2. Use below table to label each action.
    
| label |  #0  |    #1     |     #2     |     #3     |    #4     |     #5     |     #6     |     #7     |     #8     |     #9     |    #10    |    #11    |
|------ | ---- | --------- | ---------- |----------- |---------- |----------- |----------- |----------- |----------- |----------- |-----------|-----------|
|Action | Null | Pattern 1 | Pattern 2A | Pattern 2B | Pattern 3 | Pattern 4A | Pattern 4B | Pattern 5A | Pattern 5B | Pattern 6A | Pattern 6B| Full HW   |			 


Data Preparation Steps.

3. Split the data into train, val, test (Make sure each action is ~balanced in training dataset).
    * ..../data/train/
    * ..../data/val/
    * ..../data/test/
4. Window data using [create_window.py]
    * go to scripts/data_preparation/
    * python create_windows.py -s PATH_TO_DATA\train\ -d PATH_TO_SAVE\train\ -a "0,1,2,3,4,5,6,7,8,9,10" -v "new" -c 6 -m 0 -x 1
5. Optional some dataset collected using older version of software has duplicated data points, to remove them use [drop_duplicate_rows.py]

### Model Training & Inference
1. Model training
    * go to [train_params.json] and specify the required parameters like in example. 
    * run the cmd, python train.py
2. Inferencing 
    * go to [inference_params.json] and specify the required parameters like in example.
    * run the cmd, python inference.py

