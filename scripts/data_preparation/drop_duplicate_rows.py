# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 18:24:51 2020

@author: Mathanraj-Sharma
@email: rvmmathanraj@gmail.com
"""

import pandas as pd
import os
import argparse


read_from = "F:/intern/senzanalytics-handwash_recognition-cc375ddb4bbc/senzanalytics-handwash_recognition-cc375ddb4bbc/data/data-collection-23-09-2020/"
write_to = "F:/intern/senzanalytics-handwash_recognition-cc375ddb4bbc/senzanalytics-handwash_recognition-cc375ddb4bbc/data/data-collection-23-09-2020/no_duplicates/"
column = "count"
    
if not os.path.exists(write_to):
   os.makedirs(write_to)
   
file_names = [f.name for f in os.scandir(read_from) if f.is_file()]


# drop duplicate rows and save csv in destination folder
for file in file_names:
    df = pd.read_csv(read_from+file)
    try:
        df.drop_duplicates(subset=column, keep='first', inplace=True)
    except KeyError:
        pass
    df.to_csv(write_to+file, index=False)

print("Finished and files are saved to "+write_to)

# =============================================================================
#     python drop_duplicate_rows.py -s ..\..\data\data-collection-23-09-2020\ -d ..\..\data\data-collection-23-09-2020\no_duplicates\ -c "count"
# =============================================================================
