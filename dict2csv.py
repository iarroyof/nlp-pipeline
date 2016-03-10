import csv
import os

filename = "svr_output_headlines_100_d2v_convs_300_m5.txt"
csv_file = "svr_output_headlines_100_d2v_convs_300_m5.csv"
columns = ['kernel','C','degree', 'gamma', 'best_score','best_params']
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
from pdb import set_trace as st()
with open(filename) as f:
    for i, line in enumerate(f):
        try:
            dict_d = { key: eval(line.strip())[key] for key in csv_columns }
            dict_data.append(dict_d)
        except KeyError:
            pass

currentPath = os.getcwd()
csv_file = currentPath + "/csv/"+ csv_file

WriteDictToCSV(csv_file,csv_columns,dict_data)
