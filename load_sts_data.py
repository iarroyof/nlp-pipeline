#Import libraries, prepare verbose and set options
import time
import os
import random
import re
import codecs
import numpy as np
from collections import Counter


def verbose(*args):
    print " ".join([str(a) for a in args])

class Opts:
    verbose=False
    filter_test=".*"

opts=Opts()
from pdb import set_trace as st
def load_phrases_from_file(dirname,filename,format='2017',translation=False):
    if format != "validation":
        re_file=re.compile('.*\.input\..*\.txt$')
    else:
        re_file=re.compile('.*\.input\..*en-en\.txt$')
        

    if translation:
        re_file_translation=re.compile('.*\.input\..*\.translation.txt$')
        re_file=re_file_translation
    
    phrases=[]
    if not re_file.match(filename):
        return []
    
    with codecs.open(os.path.join(dirname,filename),encoding='utf-8') as data:
        for line in data:
            bits=line.strip().split('\t')
            if len(bits)>=2 or len(bits)<=4:
                if not format or format == "validation":
                    phrases.append((bits[0],bits[1]))
                    
                        
                elif format=="2017":
                    phrases.append((bits[2],bits[3]))
    return phrases

def load_gs_from_file(dirname,filename):
    re_gs=re.compile('.*\.gs\..*\.(txt|ascii)$')
    gs=[]
    if not re_gs.match(filename):
        return []

    with open(os.path.join(dirname,filename)) as data:
        for line in data:
            line=line.strip()
            try:
                gs.append(float(line))
            except ValueError:
                gs.append(0.0)
    return gs

def load_all_phrases(dirname,filter=".input.",format=None,translation=False):
    all_phrases=[]
    filter_dirs=re.compile(filter)
    for filename in os.listdir(dirname):
        if not filter_dirs.search(filename):
            continue
        phrases=load_phrases_from_file(dirname,filename,format=format,translation=translation)
        if len(phrases)>0:
            all_phrases.append((filename,phrases))
    return all_phrases

def load_all_gs(dirname):
    all_gs=[]
    for filename in os.listdir(dirname):
        gs=load_gs_from_file(dirname,filename)
        if len(gs)>0:
            all_gs.append((filename,gs))
    return all_gs

def load_train_dirs(dirs):
    train_data=[]
    gs_data=[]
    for directory,format,translation in dirs:
        verbose('Starting training')
        train_data_=load_all_phrases(os.path.join(directory,''),format=format,translation=translation)
        
        if format != "validation":
            gs_data_=dict(load_all_gs(os.path.join(directory,'')))

        for (n,d) in train_data_:
            n_=n.replace('input', 'gs')
            if translation:
                n_=n_.replace('.translation', '')
            for i,s in enumerate(d):
                train_data.append(s[0].encode('utf-8'))
                train_data.append(s[1].encode('utf-8'))
                if format != "validation":
                    gs_data.append(gs_data_[n_][i])
            if format != "validation":
                verbose("Phrases in",n,len(d),len(gs_data_[n_]))
            else:
                verbose("Phrases in",n,len(d))
        verbose('Total train phrases',directory,sum([len(d) for n,d in train_data_]))

        verbose('Total train phrases',len(train_data))
    return train_data,gs_data
