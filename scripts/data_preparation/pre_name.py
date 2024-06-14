import os
import pandas as pd

def rename(read_from,write_to):
    file_names = [f.name for f in os.scandir(read_from) if f.is_file()]
    
    for name in file_names:
        oldname = name
        if '_'+'14'+'.csv' in name:
            newname=oldname[:-6]+'1'+'.csv'
            os.rename(read_from + name, write_to + newname)
            df = pd.read_csv(write_to + newname)
            df["label_0"] = df["label_0"].replace({14:1})
            df.to_csv(write_to + newname)
        elif '_'+'15'+'.csv' in name:
            newname=oldname[:-6]+'1'+'.csv'
            os.rename(read_from + name, write_to + newname)
            df = pd.read_csv(write_to + newname)
            df["label_0"] = df["label_0"].replace({15:1})
            df.to_csv(write_to + newname)
        elif '_'+'16'+'.csv' in name:
            newname=oldname[:-6]+'9'+'.csv'
            os.rename(read_from + name, write_to + newname)
            df = pd.read_csv(write_to + newname)
            df = pd.read_csv(write_to + newname)
            df["label_0"] = df["label_0"].replace({16:9})
            df.to_csv(write_to + newname)
        else:
            newname=oldname[:-6]+'10'+'.csv'
            os.rename(read_from + name, write_to + newname)
            df = pd.read_csv(write_to + newname)
            df = pd.read_csv(write_to + newname)
            df["label_0"] = df["label_0"].replace({17:10})
            df.to_csv(write_to + newname)
         


if __name__ == "__main__":

    read_from = "F:/intern/senzanalytics-handwash_recognition-cc375ddb4bbc/senzanalytics-handwash_recognition-cc375ddb4bbc/data/data-collection-23-09-2020/Thilaksi_280321_R/"
    write_to = "F:/intern/senzanalytics-handwash_recognition-cc375ddb4bbc/senzanalytics-handwash_recognition-cc375ddb4bbc/data/data-collection-23-09-2020/ThilaksiR/"
    rename(read_from,write_to)
