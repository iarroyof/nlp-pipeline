import csv
import os
import numbers
filename = "svr_output_headlines_100_d2v_convs_300_m5.txt"
csv_file = filename + ".csv"
columns = ['kernel','C','degree', 'gamma', 'performance']
ord = 4
from pdb import set_trace as st
def _finditem(obj, key):
    if key in obj:
        v = obj[key] 
        if isinstance(v, numbers.Real):
            return round(v, ord)
        else:
            return v
    for k, v in obj.items():
        if isinstance(v,dict):
            item = _finditem(v, key)
            if item is not None:
                if isinstance(item, numbers.Real):
                    return round(item, ord)
                else:
                    return item
            else:
                return "N/A"
                    
def WriteDictToCSV(csv_file,csv_columns,dict_data):
    try:
        with open(csv_file, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            writer.writeheader()
            for data in dict_data:
                writer.writerow(data)
    except IOError as (errno, strerror):
            #print("I/O error({0}): {1}".format(errno, strerror))    
            pass
    return            

csv_columns = columns

dict_data = []
from pdb import set_trace as st

with open(filename) as f:
    for i, line in enumerate(f):
        try:
            #st()
            dict_d = { key: _finditem(eval(line.strip()), key) for key in csv_columns }
            dict_data.append(dict_d)
        except KeyError:
            pass

currentPath = os.getcwd()
csv_file = currentPath + "/csv/"+ csv_file

WriteDictToCSV(csv_file,csv_columns,dict_data)
