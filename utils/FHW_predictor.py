# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 16:22:04 2020

@author: Mathanraj-Sharma
"""
import os
import numpy as np
from sklearn.metrics import accuracy_score
import pandas as pd

class FHW_predictor():
    
    def __init__(self, model, data_processor, output_dir):
        """
        Create an instance of FHW_predictor.

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
        Instance/Object of FHW_predictor.

        """
        self.model = model
        self.data_processor = data_processor
        self.output_dir = output_dir
        
        
    def FHW_logic(self, predictions, classes=7, expected=1, consider=2, thresh=5, ordered=False):
        """
        Decide whether a prediction list is Full handwash or not.

        Parameters
        ----------
        predictions : list/iterable
            Iterable of predicted labels.
        classes : int, optional
            Number of classes to consider for FHW decision. The default is 7.
        expected : int, optional
            Number of time each action identified. The default is 1.
        consider : int, optional
            Number of time a label should come sequentially. The default is 2.
        thresh : int, optional
            Number of actions counted at least "expected" time. The default is 5.
        ordered : boolean, optional
            Consider actions should be predicted in order or not. The default is False.

        Returns
        -------
        boolean
            Given list of predictions is Full handwash or not.

        """
        
        # only selected 7 actions will be counted
        if classes == 7:
            counter = { 1: 0, 2: 0, 3: 0, 5: 0, 6: 0, 7: 0, 8: 0 }
            correct = 7
            thresh = thresh
        # all actions will be counted
        else:
            counter = { 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0 }
            correct = 10
            thresh = thresh
            
        last_occur = 0
        
        if ordered:
            for i in range(2, len(predictions)):
                # one action should be predicted three times in sequence (one after the other)
                if (consider == 3) and (predictions[i] in counter.keys()) and (predictions[i] == predictions[i-1]) and (predictions[i] == predictions[i-2]):
                    counter[predictions[i]] += 1 if (predictions[i] >= last_occur) else 0
                    last_occur = predictions[i]
                # one action should be predicted two times in sequence (one after the other)
                elif (consider == 2) and (predictions[i] in counter.keys()) and (predictions[i] == predictions[i-1]):
                    counter[predictions[i]] += 1 if (predictions[i] >= last_occur) else 0
                    last_occur = predictions[i]
                # one action should be predicted one time in sequence 
                elif (consider == 1) and (predictions[i] in counter.keys()):
                    counter[predictions[i]] += 1 if (predictions[i] >= last_occur) else 0
                    last_occur = predictions[i]
                else:
                    continue
        else:        
            for i in range(2, len(predictions)):
                if (consider == 3) and (predictions[i] in counter.keys()) and (predictions[i] == predictions[i-1]) and (predictions[i] == predictions[i-2]):
                    counter[predictions[i]] += 1
                elif (consider == 2) and (predictions[i] in counter.keys()) and (predictions[i] == predictions[i-1]):
                    counter[predictions[i]] += 1
                elif (consider == 1) and (predictions[i] in counter.keys()):
                    counter[predictions[i]] += 1
                else:
                    continue
                
        # check whether every action predicted at least expected time
        for key, value in counter.items():
            if value < expected:
                correct -= 1
        
        return True if correct >= thresh else False
    
    
    def FHW_custom_score(self, path, **kwargs):
        """
        Calculate score for full handwash data provided. 

        Parameters
        ----------
        path : str
            Path to read windowed full handwash data.
        **kwargs : dict
            Arguments need to be passed for FHW_logic function.

        Returns
        -------
        results : dict
            FHW_logic parameteras and their result summaries.

        """
        
        def Diff(li1, li2):
            """
            Indentify different elements in two lists.
            """
            li_dif = [i for i in li1 + li2 if i not in li1 or i not in li2]
            return li_dif
        
        case_names = [f.name.split('_')[0] for f in os.scandir(path) if f.is_file()]
        case_names = sorted(set(case_names))
        
    #     print(case_names)
        fhw_count_7 = 0
        fhw_count_10 = 0
        true_7 = []
        true_10 = []
        total_fhw = len(case_names)
        
        for test_case_name in case_names:
            Xp_test, yp_test = self.data_processor.prepare_for_inference(path, test_case_name, add_zero=False, encode_labels=False)
            # true_labels = np.argmax(yp_test, axis=1)
            
            if kwargs['prob_threshed']:
                pred_proba = self.model.predict(Xp_test)
                predictions = [np.argmax(p) if max(p) > kwargs['p_thresh'] else 0 for p in pred_proba]
            else:
                predictions = np.argmax(self.model.predict(Xp_test), axis=-1)
            
            if self.FHW_logic(predictions, classes=7, expected=kwargs['expected'], consider=kwargs['consider'], thresh=kwargs['thresh'], ordered=kwargs['ordered']):
                fhw_count_7 += 1
                true_7.append(test_case_name)
            
            if self.FHW_logic(predictions, classes=10, expected=kwargs['expected'], consider=kwargs['consider'], thresh=kwargs['thresh'], ordered=kwargs['ordered']):
                fhw_count_10 += 1
                true_10.append(test_case_name)
                
        results = {'expected': kwargs['expected'], 'consider': kwargs['consider'], 'p_thresh': kwargs['p_thresh'],
                  'thresh': kwargs['thresh'], 'ordered': kwargs['ordered'], 
                  'classes_7': (fhw_count_7/total_fhw) * 100, 
                  'classes_10': (fhw_count_10/total_fhw) * 100}
        
        if kwargs['save']:
            # print('Saving...')
            with open(self.output_dir+f"/{kwargs['save_to']}.txt", 'w') as file:
                file.write(f'fhw_7: {fhw_count_7}\n')
                file.write(f'fhw_10: {fhw_count_10}\n')
                file.write(f'total_fhw: {total_fhw}\n')
                file.write(f'Not Selected in 7 class: {Diff(case_names, true_7)}\n')
                file.write(f'Not Selected in 10 class: {Diff(case_names,true_10)}\n')
                file.write(f"if 7 classes considered and expected {kwargs['thresh']} to be correct = {(fhw_count_7/total_fhw) * 100} \n")
                file.write(f"if 10 classes considered and expected {kwargs['thresh']} to be correct = {(fhw_count_10/total_fhw) * 100} \n")
            # print('Saved Successfully..!')
        return results
    
            
    def record_FHW_predictions(self, path, file_name, encode_labels=False):
        """
        Predict and save fhw data in excel file. 

        Parameters
        ----------
        path : str
            Path to read windowed full handwash data.
        file_name : str
            Name of file to be saved in output_directory.
        encode_labels : boolean, optional
            Apply or not to apply OneHotEncoding for the labels. The default is False.

        Returns
        -------
        None.

        """
        case_names = [f.name.split('_')[0] for f in os.scandir(path) if f.is_file()]
        case_names = sorted(set(case_names))
        fhw_dict = dict()
        
        # print(case_names)
        for test_case_name in case_names:
            Xp_test, yp_test = self.data_processor.prepare_for_inference(path, test_case_name, add_zero=False, encode_labels=encode_labels)
            predictions = np.argmax(self.model.predict(Xp_test), axis=-1)
            true_labels = np.argmax(yp_test, axis=1)
            
            acc = accuracy_score(true_labels, predictions)        
            
            fhw_dict[test_case_name] = pd.DataFrame({
                'true' : true_labels,
                'pred': predictions,
                'acc': acc
            })
            
        with pd.ExcelWriter(self.output_dir+f'{file_name}.xlsx', engine='openpyxl', mode='w') as writer:
            for title, df in fhw_dict.items():
                df.to_excel(writer, sheet_name=title)
            writer.save()
        writer.close()
        
        
    def record_FHW_scores(self, path, file_name, save=False, p_thresh_list = [8,85,95], thresh_list=[5,6,7], consider_list=[1,2]):
        """
        Predicte and save full handwash scores for given set of data.

        Parameters
        ----------
        path : str
            Path to read windowed full handwash data.
        save : boolean, optional
            whether to save full handwash scores in txt file or not. The default is False.
        p_thresh_list : list, optional
            List of interger values for threshold softmax outputs. The default is [8,85,95].
        thresh_list : list, optional
            List of interger for how many time each action should be counted. The default is [5,6,7].
        consider_list : list, optional
            List of intergerd how many time each action should be predicted sequentially. The default is [1,2].

        Returns
        -------
        None.

        """
        temp = []
        for i in p_thresh_list:
            for j in thresh_list:
                for k in consider_list:
                    temp.append(self.FHW_custom_score(path, save_to=f'FHW_pred_consider_{k}_pthresh_{i}', expected=1, consider=k, thresh=j, prob_threshed=True, p_thresh= i/100, save=save, ordered=False))
                    temp.append(self.FHW_custom_score(path, save_to=f'FHW_pred_consider_{k}_pthresh_{i}_ordered', expected=1, consider=k, thresh=j, prob_threshed=True, p_thresh= i/100, save=save, ordered=True))
            
        df = pd.DataFrame(temp)
        df['p_thresh'] = (df['p_thresh'] * 100).astype(int)
        df[['classes_7', 'classes_10']] = df[['classes_7', 'classes_10']].round(2)
        df.sort_values(['classes_7', 'classes_10'], inplace=True)
        df_75 = df[df['classes_7'] > 75]
        
        results = {'df': df, 'df_75': df_75}
        with pd.ExcelWriter(self.output_dir+f'{file_name}.xlsx', engine='openpyxl', mode='w') as writer:
            for title, df in results.items():
                df.to_excel(writer, sheet_name=title, index=False)
            writer.save()
        writer.close()
            
        