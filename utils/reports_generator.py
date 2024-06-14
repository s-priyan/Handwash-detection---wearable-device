# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 10:53:45 2020

@author: Mathanraj-Sharma
"""

import os
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report

class reports_generator():
    
    def __init__(self, model, data_processor, output_dir):
        """
        Create an object of reports_generator class.

        Parameters
        ----------
        model : BASE
            Any model specified under classifiers pockage.
        data_processor : data_processor
            Instance of data_processor (transformed) class.
        output_dir : str
            Path to save outputs.

        Returns
        -------
        reports_generator instance.

        """
        self.model = model
        self.data_processor = data_processor
        self.output_dir = output_dir
        self.cols = ['Natural movement', 'Pattern 1', 'Pattern 2 A','Pattern 2 B',
                     'Pattern 3', 'Pattern 4 A','Pattern 4 B','Pattern 5 A', 'Pattern 5 B', 'Pattern 6 A', 'Pattern 6 B']
        
    
    def get_confusion_matrix(self, X_test, y_test, p_thresh=None):
        """
        Predict and construct confusion matrix with specified threshold for softmax output.

        Parameters
        ----------
        X_test : np.ndarray
            Array of testing features.
        y_test : np.ndarray
            OneHotEncoded True Labels of testing data.
        p_thresh : float, optional
            Threshold value for softmax output. The default is None.

        Returns
        -------
        cmtxp : pd.DataFrame
            Confusion matrix for testing data.

        """
        pred_proba = self.model.predict(X_test)
        
        if p_thresh:
            pred_thresh = [np.argmax(p, axis=-1) if max(p) > p_thresh else 0 for p in pred_proba]
        else:
            pred_thresh = np.argmax(pred_proba, axis=-1)
            
        # print(pred_thresh)
        y_test = np.argmax(y_test, axis=1)
        # print(y_test)
        
        unique_label = np.unique([y_test, pred_thresh])
        cmtxp = pd.DataFrame(
            confusion_matrix(y_test, pred_thresh, labels=unique_label), 
            index=['true:{:}'.format(x) for x in unique_label], 
            columns=['pred:{:}'.format(x) for x in unique_label]
        )
    
        cmtxp.columns = self.cols
        cmtxp.index = self.cols
    
        cmp = cmtxp.to_numpy().astype('float64') / cmtxp.to_numpy().sum(axis=1)[:, np.newaxis]
        cmtxp['Class Accuracy'] = np.around(cmp.diagonal() * 100, 2)
        
        return cmtxp
    
    
    def get_classification_report(self, X_test, y_test, p_thresh=None):
        """
        Generate and return classification report and confusion matrix for testing data. 

        Parameters
        ----------
        X_test : np.ndarray
            Array of testing features.
        y_test : np.ndarray
            OneHotEncoded True Labels of testing data.
        p_thresh : float, optional
            Threshold value for softmax output. The default is None.

        Returns
        -------
        list
            List containing the pd.DataFrame objects of classification report and confusion matrix.

        """
        pred_proba = self.model.predict(X_test)
        
        if p_thresh:
            pred_thresh = [np.argmax(p) if max(p) > p_thresh else 0 for p in pred_proba]
        else:
            pred_thresh = np.argmax(pred_proba, axis=1)
            
        
        cls_report = pd.DataFrame(classification_report(np.argmax(y_test, axis=1), pred_thresh, target_names=self.cols, output_dict=True)).transpose()
        cm = self.get_confusion_matrix(X_test, y_test, p_thresh=p_thresh)
        
        return [cls_report, cm]
    
    
    def generate_classification_reports(self, X_test, y_test, p_thresh_list=[1, 80, 85, 90, 95]):
        """
        Generate and save classification reports for list of softmax threshold values.
        Reprots will be written into specified output directory with name classification_reports.xlsx.
        
        Parameters
        ----------
        X_test : np.ndarray
            Array of testing features.
        y_test : np.ndarray
            OneHotEncoded True Labels of testing data.
        p_thresh_list : list, optional
            list containing int threshold values for softmax output. The default is [1, 80, 85, 90, 95].

        Returns
        -------
        None.

        """
        cm_cr = dict()
        
        for i in p_thresh_list:
            cm_cr[f'classification-report-thresh-{i}'] = self.get_classification_report(X_test, y_test, p_thresh=(i/100))
        
        with pd.ExcelWriter(self.output_dir+'/classification-report.xlsx', engine='openpyxl', mode='w') as writer:
            for title, dfs in cm_cr.items():
                dfs[0].to_excel(writer, sheet_name=title, float_format="%.3f")
                dfs[1].to_excel(writer, sheet_name=title, float_format="%.3f", startrow=25)
            writer.save()
        writer.close()            
            
                
    
    def create_prob_summary(self, X_test, y_test):
        """
        Gather top 3 softmax outputs for every window in testing data and summarize it.

        Parameters
        ----------
        X_test : np.ndarray
            Array of testing features.
        y_test : np.ndarray
            OneHotEncoded True Labels of testing data.

        Returns
        -------
        d_top : dict
            Dictionary containing .   
            'true_label',
            'top_1 prediction',
            'top_1P probability',
            'top_2 prediction',
            'top_2P probability,
            'top_3 prediction',
            'top_3P probability',
            
            'if top_X == true_label'
            'top_1_true',
            'top_2_true',
            'top_3_true'.

        """
        prediction = self.model.predict(X_test)
        # sorting the predictions in descending order
        sorting = (-prediction).argsort()
    
        top_1 = []
        top_2 = []
        top_3 = []
        top_1P = []
        top_2P = []
        top_3P = []
    
        for i in range(len(prediction)):
            # getting the top 3 predictions
            sorted_ = sorting[i][:3]
            t = 1
            for value in sorted_:
    
                # get predicted label for top 3 probs
                predicted_label = [int(x) for x in self.data_processor.one_hot_encoder.categories_[0]][value]
                prob = (prediction[i][value]) * 100
                prob = "%.2f" % round(prob,3)
    
                if t == 1:
                    top_1.append(predicted_label)
                    top_1P.append(prob)
                    t = 2
                elif t == 2:
                    top_2.append(predicted_label)
                    top_2P.append(prob)
                    t = 3
                elif t == 3:
                    top_3.append(predicted_label)
                    top_3P.append(prob)
       
        d_top = pd.DataFrame({
            'true_label': np.argmax(y_test, axis=1),
            'top_1': top_1,
            'top_1P': top_1P,
            'top_2': top_2,
            'top_2P': top_2P,
            'top_3': top_3,
            'top_3P': top_3P,
        })
        
        d_top['top_1_true'] = np.where(d_top['true_label'] == d_top['top_1'], 1, 0)
        d_top['top_2_true'] = np.where(d_top['true_label'] == d_top['top_2'], 1, 0)
        d_top['top_3_true'] = np.where(d_top['true_label'] == d_top['top_3'], 1, 0)
        
        return d_top
    
    def top_percentages(self, d_top):
        """
        Calculate percentage of correctly predicted labels in each top_1, top_2 and top_3 probabilities.

        Parameters
        ----------
        d_top : dict
            Dictionary of top 3 softmax output summary.

        Returns
        -------
        temp : dict
            Percentage of correctly predicted labels in each category.

        """
        top_1 = []
        top_2 = []
        top_3 = []
        for i in [*range(0,11)]:
            temp = d_top[d_top['true_label'] == i]
            top_1.append((temp['top_1_true'].sum()/temp.shape[0])*100)
            top_2.append((temp['top_2_true'].sum()/temp.shape[0])*100)
            top_3.append((temp['top_3_true'].sum()/temp.shape[0])*100)
    
    
        temp = pd.DataFrame({
            'true_label':[*range(0,11)],
            'top_1 %':top_1,
            'top_2 %':top_2,
            'top_3 %':top_3,
        })
        
        return temp
    
    def generate_individual_report(self, path, p_thresh=None):
        """
        Generate and save confusion_matrix, probability summary and probabilty percentage
        for each individual person in testing data.
        
        Outputs will be saved in output directory with names, confusion-matrix.xlsx and prob-summary.xlsx. 

        Parameters
        ----------
        path : str
            Windowed testing data folder path.
        p_thresh : float, optional
            probabilty summary to consider when constructing confusion matrix. The default is None.

        Returns
        -------
        None.

        """
        case_names = [f.name.split('_')[0] for f in os.scandir(path) if f.is_file()]
        case_names = sorted(set(case_names))
        # print(case_names)
        
        # initialize empty dictionatries
        cm_dict = dict()
        prob_dict = dict()
        per_dict = dict()
        
        for test_case_name in case_names:
            Xp_test, yp_test = self.data_processor.prepare_for_inference(path, test_case_name)
            
            # predictions = np.argmax(self.model.predict(Xp_test), axis=-1)
    
            cm = self.get_confusion_matrix(Xp_test, yp_test, p_thresh=p_thresh)
            cm_dict[test_case_name] = cm.copy()
            
            d_top = self.create_prob_summary(Xp_test, yp_test)
            prob_dict[test_case_name] = d_top.copy()
            
            top_per = self.top_percentages(d_top)
            per_dict[test_case_name] = top_per.copy()
        
        
        # write the confusion matrix and probability summaries to xlsx
        with pd.ExcelWriter(self.output_dir+'/confusion-matrix.xlsx', engine='openpyxl', mode='w') as writer:
            for title, df in cm_dict.items():
                df.to_excel(writer, sheet_name=title)
            writer.save()
        writer.close()
        
        #  write top probabilty summary to excel sheet
        with pd.ExcelWriter(self.output_dir+'/prob-summary.xlsx', engine='openpyxl', mode='w') as writer:
            for title, df in prob_dict.items():
                df.to_excel(writer, sheet_name=title, float_format="%.3f")
                
            for title, df in per_dict.items():
                 df.to_excel(writer, sheet_name=title, float_format="%.3f", startcol=14, index=False)
            writer.save()
        writer.close()
        