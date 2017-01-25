#Import libraries, prepare verbose and set options
import time
import os
import random
import re
import codecs
import numpy as np
from collections import Counter

YEAR="2013"
MAX_NB_WORDS=20000
MAX_SEQUENCE_LENGTH=50
VALIDATION_SPLIT=0.30
GLOVE_DIR="/almac/ignacio/data/glove"
EMBEDDING_DIM=300
TRAIN_DIRS=[
    ("/almac/ignacio/data/sts_all/train-"+YEAR,None,False)]

def verbose(*args):
    print " ".join([str(a) for a in args])

class Opts:
    verbose=False
    filter_test=".*"

opts=Opts()

def load_phrases_from_file(dirname,filename,format='2017',translation=False):
    re_file=re.compile('.*\.input\..*\.txt$')
    re_file_translation=re.compile('.*\.input\..*\.translation.txt$')

    if translation:
        re_file=re_file_translation
    
    phrases=[]
    if not re_file.match(filename):
        return []

    with codecs.open(os.path.join(dirname,filename),encoding='utf-8') as data:
        for line in data:
            bits=line.strip().split('\t')
            if len(bits)>=2 or len(bits)<=4:
                if not format:
                    try:
                        phrases.append((bits[0],bits[1]))
                    except:
                        st()
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
        gs_data_=dict(load_all_gs(os.path.join(directory,'')))



        for (n,d) in train_data_:
            n_=n.replace('input', 'gs')
            if translation:
                n_=n_.replace('.translation', '')
            for i,s in enumerate(d):
                train_data.append(s[0].encode('utf-8'))
                train_data.append(s[1].encode('utf-8'))
                gs_data.append(gs_data_[n_][i])
            verbose("Phrases in",n,len(d),len(gs_data_[n_]))
        verbose('Total train phrases',directory,sum([len(d) for n,d in train_data_]))

        verbose('Total train phrases',len(train_data))
    return train_data,gs_data


train_data_,gs_data=load_train_dirs(TRAIN_DIRS)

print "Avg size:",np.mean([len(x.split()) for x in train_data_])
print "Max size:",np.max([len(x.split()) for x in train_data_])
print "Min size:",np.min([len(x.split()) for x in train_data_])

train_data=[]
for i in range(len(train_data_)/2):
    train_data.append(train_data_[i*2]+" ### "+train_data_[i*2])

print "Avg size after merging:",np.mean([len(x.split()) for x in train_data])
print "Max size after merging:",np.max([len(x.split()) for x in train_data])
print "Min size after merging:",np.min([len(x.split()) for x in train_data])
print "Total examples",len(train_data)

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical

tokenizer = Tokenizer(nb_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(train_data)
sequences = tokenizer.texts_to_sequences(train_data)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

data = pad_sequences(sequences, maxlen=2*MAX_SEQUENCE_LENGTH)

labels = to_categorical(np.asarray(gs_data))
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)

# split the data into a training set and a validation set
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

x_train = data[:-nb_validation_samples]
y_train = labels[:-nb_validation_samples]
x_val = data[-nb_validation_samples:]
y_val = labels[-nb_validation_samples:]

print('Shape of train:',x_train.shape)
print('Shape of train:',y_train.shape)
print('Shape of train:',x_val.shape)
print('Shape of train:',y_val.shape)

embeddings_index = {}
f = open(os.path.join(GLOVE_DIR, 'glove.6B.%dd.txt' % EMBEDDING_DIM))
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))
embeddings_index['###'] = np.zeros(100)

embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector
print embedding_matrix

from keras.layers import Embedding, Activation
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Model
from keras.models import Sequential
from keras.layers import LSTM,Bidirectional

embedding_layer = Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH*2,
                             dropout=0.2,
                            trainable=False)

model = Sequential()
model.add(embedding_layer)
model.add(Bidirectional(LSTM(300, dropout_W=0.2, dropout_U=0.2)))
#model.add(Dense(200))
model.add(Dense(80))
model.add(Dense(6, activation='softmax'))
#model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

print(model.summary())

# happy learning!
model.fit(x_train, y_train, validation_data=(x_val, y_val),
          nb_epoch=50, batch_size=20)

