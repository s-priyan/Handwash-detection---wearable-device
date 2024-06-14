# -*- coding: utf-8 -*-
"""
Created on Sun Nov 29 13:27:22 2020

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


if __name__ == '__main__':
    
    with open('./inference_params.json', 'r') as file:
        args = json.load(file)
    
    file_name = f"{args['model']}-{args['axis']}Axis-{args['hand_combo']}-{args['hand_info']}-{args['other']}-"
    date = datetime.now().strftime("%d-%m-%Y")
    output_directory = f"{args['output_dir']}/{date}/INFERENCES-pre_window-{file_name}"+datetime.now().strftime("%d-%m-%Y-%H-%M-%S")+'/'
    
    # load and preprocess train and test data
    print('Creating data_processor and processing training and testing data...')
    data_pro = data_processor(args['input_dir'], output_dir=output_directory)
    data_pro.fit_transform()
    
    # Load saved best model after training
    print('Loading trained best model...')
    model = keras.models.load_model(f"{args['model_path']}", compile=False)
    
#    # generate reports
#    print('Creating reports_generator object and generating reports...')
#    rg = reports_generator(model, data_pro, output_dir=output_directory)
#    rg.generate_classification_reports(data_pro.X_test, data_pro.y_test)
#    rg.generate_individual_report(args['input_dir']+'/test/')
#    
#    #  generate full handwash reports
#    print('Creating FHW_predictor object and generating FHW predictions...')
#    fhw = FHW_predictor(model, data_pro, output_dir=output_directory)
#    
#    # FHW score for old data
#    if args['fhw_nolabeled']:
#        fhw.record_FHW_predictions(args['fhw_nolabeled_dir'] ,'FHW_predictions_nolabeled', encode_labels=False)
#        results = fhw.record_FHW_scores(args['fhw_nolabeled_dir'], file_name='FHW_score_nolabeled')
#    
#    # FHW score for new labeld data
#    if args['fhw_labeled']:
#        fhw.record_FHW_predictions(args['fhw_labeled_dir'] ,'FHW_predictions_labeled', encode_labels=True)
#        results = fhw.record_FHW_scores(args['fhw_labeled_dir'], file_name='FHW_score_labeled' )
#    
#    print(f"Completed inferencing and summaries stored at {output_directory}")
    
    test_data = data_pro.X_test
    
    prediction = model.predict(test_data)
    
    label_list=[]
    for value in prediction:
        for idx in value:
            list_value=list(value)
            label=list_value.index(max(list_value))
            label_list.append(label)
            
    count_value={}
    
    for idx in range(11):
        count_value[idx]=label_list.count(idx)