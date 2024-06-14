# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 22:55:07 2020

@author: Mathanraj-Sharma
@email: rvmmathanraj@gmail.com
"""

import sys  
sys.path.append('../') 

from classifiers import *
from utils import data_processor
from utils import reports_generator
from utils import FHW_predictor
from datetime import datetime
import tensorflow.keras as keras
import json
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

def get_model(model_type, out_dir, kwargs):
    
    if model_type == 'BASE':
        return BASE(output_dir=out_dir, **kwargs)
    elif model_type == 'Simple_CNN':
        return SimpleCNN(output_dir=out_dir, **kwargs)
    elif model_type == 'Multi_Headed_CNN':
        return Multi_Headed_CNN(output_dir=out_dir, **kwargs)
    elif model_type == 'FCN':
        return FCN(output_dir=out_dir, **kwargs)
    elif model_type == 'RESNET':
        return RESNET(output_dir=out_dir, **kwargs)
    elif model_type == 'CNN_LSTM':
        return CNN_LSTM(output_dir=out_dir, **kwargs)
    elif model_type == 'Bi_LSTM':
        return Bi_LSTM(output_dir=out_dir, **kwargs)
    elif model_type == 'LSTM':
        return LSTM(output_dir=out_dir, **kwargs)
    elif model_type == 'GRU':
        return GRU(output_dir=out_dir, **kwargs)
    else:
        raise ValueError('Unknown model specified in params.json')
        
if __name__ == "__main__":
    
    with open('./train_params.json', 'r') as file:
        args = json.load(file)
    
    file_name = f"{args['model_type']}-{args['axis']}Axis-{args['hand_combo']}-{args['hand_info']}-{args['other']}-"
    date = datetime.now().strftime("%d-%m-%Y")
    output_directory = f"{args['output_dir']}/{date}/pre_window-{file_name}"+datetime.now().strftime("%d-%m-%Y-%H-%M-%S")+'/'
    
    # load and preprocess train and test data
    print('Creating data_processor and processing training and testing data...')
    data_pro = data_processor(args['input_dir'], output_dir=output_directory)
    data_pro.fit_transform()
    
    # create model object and train model
    print('Creating model and start training...')
    model = get_model(args['model_type'], output_directory, args['model_kwargs'])
    model.fit(data_pro.X_train, data_pro.y_train, data_pro.X_test, data_pro.y_test)
    
    # Load saved best model after training
    print('Loading trained best model...')
    if args['model_type'] == "Multi_Headed_CNN":
        model.load_weights(f"{output_directory}/best_model.hdf5")
    else:
        model = keras.models.load_model(f"{output_directory}/best_model.hdf5", compile=False)
    
    # generate reports
    print('Creating reports_generator object and generating reports...')
    rg = reports_generator(model, data_pro, output_dir=output_directory)
    rg.generate_classification_reports(data_pro.X_test, data_pro.y_test)
    rg.generate_individual_report(args['input_dir']+'/test/')
    
    #  generate full handwash reports
    print('Creating FHW_predictor object and generating FHW predictions...')
    fhw = FHW_predictor(model, data_pro, output_dir=output_directory)
    
    # FHW score for old data
    if args['fhw_nolabeled']:
        fhw.record_FHW_predictions(args['fhw_nolabeled_dir'] ,'FHW_predictions_nolabeled', encode_labels=False)
        results = fhw.record_FHW_scores(args['fhw_nolabeled_dir'], file_name='FHW_score_nolabeled')
    
    # FHW score for new labeld data
    if args['fhw_labeled']:
        fhw.record_FHW_predictions(args['fhw_labeled_dir'] ,'FHW_predictions_labeled', encode_labels=True)
        results = fhw.record_FHW_scores(args['fhw_labeled_dir'], file_name='FHW_score_labeled' )
        
    print(f"Completed model training and outputs stored at {output_directory}")
        
    
