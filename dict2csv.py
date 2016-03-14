import csv
import os
import numbers

from argparse import ArgumentParser as ap
parser = ap(description='This script converts a dictionary of results into a csv file. Probably this csv can easily be read in Calc Sheet or converted into a latex table. The output file will be stored at a previously created /csv directory. The name of the file is the same than the input file but with .csv extension.')
parser.add_argument("-i", help="Input file name. This file must contain a dictionary by line.", metavar="input_file", required=True)
parser.add_argument("-n", help="Number of digits after decimal point for real numbers.", metavar="digits_amount", default=3)
#parser.add_argument("-o", help="The operation the input data was derived from. Options: {'conc', 'convs', 'sub'}.", default="conc")
args = parser.parse_args()
filename = args.i #"svr_output_headlines_100_d2v_convs_300_m5.txt"
csv_file = filename + ".csv"
columns = ['kernel','C','degree', 'gamma', 'performance', 'best_score']
ord = args.n

def _finditem(obj, key):
    try:
        v = obj[key] 
        if isinstance(v, numbers.Real):
            return round(v, ord)
        else:
            return v
    except KeyError:
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
