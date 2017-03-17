"""
A keras attention layer that wraps RNN layers.

Based on tensorflows [attention_decoder](https://github.com/tensorflow/tensorflow/blob/c8a45a8e236776bed1d14fd71f3b6755bd63cc58/tensorflow/python/ops/seq2seq.py#L506) 
and [Grammar as a Foreign Language](https://arxiv.org/abs/1412.7449).

date: 20161101
author: wassname
url: https://gist.github.com/wassname/5292f95000e409e239b9dc973295327a
"""

# test likes in https://github.com/fchollet/keras/blob/master/tests/keras/layers/test_wrappers.py
import numpy as np
from numpy.testing import assert_allclose
from keras.utils.test_utils import keras_test
from keras.layers import wrappers, Input, recurrent, InputLayer, Merge,MaxoutDense
from keras.layers import core, convolutional, recurrent, Embedding, Dense,Flatten
from keras.models import Sequential, Model, model_from_json
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.callbacks import ModelCheckpoint
from argparse import ArgumentParser as ap


from attention_lstm_ import *
from load_sts_data import *

parser = ap()
parser.add_argument("-s", help="Number of hidden states (time steps of the moving filter)", default=10,  type=int)
parser.add_argument("-e", help="Number of training epochs", default=25, type=int)
parser.add_argument("-d", help="Number of hidden output perceptron nodes", default=20, type=int)
parser.add_argument("-m", help="Model type", default="base_line")
parser.add_argument("-D", help="Embeddings dimensions", default=100, type=int)
parser.add_argument("-t", help="Toggles train mode. Default is False.", action="store_true")
parser.add_argument("-E", help="Embedding type. 'word2vec', 'glove' 'fastText'.", default="fastText")
args = parser.parse_args()

# ----------
train=args.t
# ----------
h_STATES = args.s
DENSES = args.d
EMBEDDING_DIM = args.D
MODEL_TYPE=args.m
EPOCHS = args.e
EMBEDDING = args.E
MODEL_DIR = "/almac/ignacio"

dummy=""

if train:
    if dummy== "":
        YEARS_TRAIN=["2012", "2013", "2015", "2016"]
        #YEARS_TRAIN=["2013"]
    else:
        YEARS_TRAIN=["2013-t"]

YEAR_VALID="2017"
MAX_SEQUENCE_LENGTH=50
VALIDATION_SPLIT=0.30
VECTOR_DIR=MODEL_DIR + "/data/" + EMBEDDING + dummy
MAX_NB_WORDS=20000
params="%s_Ts%d_Ds%d_%s_H%d_Sl%d"% (MODEL_TYPE, h_STATES,DENSES, EMBEDDING,
                                                EMBEDDING_DIM, MAX_SEQUENCE_LENGTH)
model_file=MODEL_DIR + "/%s.hdf5" % params

if train:
    TRAIN_DIRS = []
    for year in YEARS_TRAIN:
        TRAIN_DIRS.append(( VECTOR_DIR.rsplit('/', 1)[0]
            + "/sts_all/train-" + year, None, False) )

VALID_DIRS=[(VECTOR_DIR.rsplit('/', 1)[0]
     + "/sts_all/valid-" + YEAR_VALID, "validation", False)]
# --------------------------
if train:
    print "Loanding train and valid dirs......"
    train_data_, gs_data=load_train_dirs(TRAIN_DIRS)

print "Loanding validation dirs......"
valid_data_, gs_test=load_train_dirs(VALID_DIRS)

print "Spliting tab-separated files..."
if train:
    train_data_A, train_data_B = train_data_[1::2], train_data_[::2]
    labels=np.asarray(gs_data)

valid_data_A, valid_data_B = valid_data_[1::2], valid_data_[::2]

#labels = to_categorical(np.asarray(gs_data))
if not train:
    test_labels=np.asarray(gs_test)
    indices_test = np.arange(test_labels.shape[0])
else:
    indices = np.arange(labels.shape[0])
    np.random.shuffle(indices)
    nb_validation_samples = int(VALIDATION_SPLIT * labels.shape[0])

    print "Labels shape: ", labels.shape

print "Tokenizing files... [A]"
tokenizer = Tokenizer(nb_words=MAX_NB_WORDS)
if train:
    tokenizer.fit_on_texts(train_data_A + valid_data_A)
    sequences_A = tokenizer.texts_to_sequences(train_data_A)
else:
    tokenizer.fit_on_texts(valid_data_A)

sequences_Av = tokenizer.texts_to_sequences(valid_data_A)
word_index_A = tokenizer.word_index
if train:
    data_A = pad_sequences(sequences_A, maxlen=MAX_SEQUENCE_LENGTH)
    data_A = data_A[indices]

x_data_Av = pad_sequences(sequences_Av, maxlen=MAX_SEQUENCE_LENGTH)

if train:
    print "Split training set into train and val... [A]"
    x_train_A = data_A[:-nb_validation_samples]
    x_val_A = data_A[-nb_validation_samples:]

print "Tokenizing files... [B]"
tokenizer = Tokenizer(nb_words=MAX_NB_WORDS)
if train:
    tokenizer.fit_on_texts(train_data_B + valid_data_B)
    sequences_B = tokenizer.texts_to_sequences(train_data_B)

else:
    tokenizer.fit_on_texts(valid_data_B)

sequences_Bv = tokenizer.texts_to_sequences(valid_data_B)
word_index_B = tokenizer.word_index
if train:
    data_B = pad_sequences(sequences_B, maxlen=MAX_SEQUENCE_LENGTH)
    data_B = data_B[indices]

x_data_Bv = pad_sequences(sequences_Bv, maxlen=MAX_SEQUENCE_LENGTH)

if train:
    print "Split training set into train and val... [B]"
    x_train_B = data_B[:-nb_validation_samples]
    x_val_B = data_B[-nb_validation_samples:]

    labels = labels[indices]
    y_train = labels[:-nb_validation_samples]
    y_val = labels[-nb_validation_samples:]

embeddings_index = {}
if EMBEDDING == "glove":
    vectors_file = VECTOR_DIR + '/glove.6B.%dd.txt' % EMBEDDING_DIM
elif EMBEDDING == "fastText":
    vectors_file = VECTOR_DIR + '/wikiEn_Full_H%d.model.vec' % EMBEDDING_DIM
elif EMBEDDING == "word2vec":
    vectors_file = VECTOR_DIR + '/w2v_En_vector_space_H%d.vec' % EMBEDDING_DIM

f = open(vectors_file)
print "Getting embedding matrix... from %s" % vectors_file
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))
#embeddings_index['###'] = np.zeros(100)

print "Filling embedding matrices..."
embedding_matrix_A = np.zeros((len(word_index_A) + 1, EMBEDDING_DIM))
for word, i in word_index_A.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix_A[i] = embedding_vector

embedding_matrix_B = np.zeros((len(word_index_B) + 1, EMBEDDING_DIM))
for word, i in word_index_B.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
       embedding_matrix_B[i] = embedding_vector

# --------------------------

def load_trained_model(model_file, model_type, vocab_a, vocab_b, timesteps, embedding_dim):

    model = build_model(model_type, vocab_a, vocab_b, timesteps, embedding_dim)
    model.load_weights(model_file)

    return model

def models(M, nb_samples, timesteps, embedding_dim):#, output_dim): # For returning sequences only

    embedding_layer = Embedding(input_dim=nb_samples + 1,
                            output_dim=embedding_dim,         
                            input_length=MAX_SEQUENCE_LENGTH, 
                            dropout=0.2,
                            trainable=False)

    if M == "base_line":
        model = Sequential()
        model.add(embedding_layer)
        model.add(Attention(recurrent.LSTM(output_dim=timesteps, return_sequences=False, consume_less='mem')))
        model.add(core.Activation('relu'))
    
    elif M == "multi_att":
        model = Sequential()
        model.add(embedding_layer)
        model.add(Attention(recurrent.LSTM(output_dim=timesteps, return_sequences=True, consume_less='mem')))
        model.add(recurrent.LSTM(output_dim=timesteps, return_sequences=True, consume_less='mem'))
        model.add(Attention(recurrent.LSTM(output_dim=timesteps, return_sequences=True, consume_less='mem')))
        model.add(recurrent.LSTM(output_dim=timesteps, return_sequences=True, consume_less='mem'))
        model.add(Attention(recurrent.LSTM(output_dim=timesteps, return_sequences=True, consume_less='mem')))
        model.add(recurrent.LSTM(output_dim=timesteps, return_sequences=True, consume_less='mem'))
        model.add(Attention(recurrent.LSTM(output_dim=timesteps, return_sequences=True, consume_less='mem')))
        model.add(recurrent.LSTM(output_dim=timesteps, return_sequences=True, consume_less='mem'))
        model.add(Attention(recurrent.LSTM(output_dim=timesteps, return_sequences=False, consume_less='mem')))
        model.add(core.Activation('relu'))

    elif M == "stacked":
    # test stacked with all RNN layers and consume_less options
        model = Sequential()
        model.add(embedding_layer)
        # model.add(Attention(recurrent.LSTM(embedding_dim, input_dim=embedding_dim,, consume_less='cpu' return_sequences=True))) # not supported
        model.add(Attention(recurrent.LSTM(output_dim=timesteps, consume_less='gpu', return_sequences=True)))
        model.add(Attention(recurrent.LSTM(output_dim=timesteps, consume_less='mem', return_sequences=True)))
        # test each other RNN type
        model.add(Attention(recurrent.GRU(output_dim=timesteps, consume_less='mem', return_sequences=True)))
        model.add(Attention(recurrent.SimpleRNN(output_dim=timesteps, consume_less='mem', return_sequences=False)))
        model.add(core.Activation('relu'))

    elif M == "stacked_1":
    # test stacked with all RNN layers and consume_less options
        model = Sequential()
        model.add(embedding_layer)
        # model.add(Attention(recurrent.LSTM(embedding_dim, input_dim=embedding_dim,, consume_less='cpu' return_sequences=True))) # not supported
        model.add(Attention(recurrent.LSTM(output_dim=timesteps, consume_less='gpu', return_sequences=True)))
        model.add(Attention(recurrent.LSTM(output_dim=timesteps, consume_less='mem', return_sequences=True)))
        # test each other RNN type
        model.add(Attention(recurrent.GRU(output_dim=timesteps, consume_less='mem', return_sequences=True)))
        model.add(Attention(recurrent.SimpleRNN(output_dim=timesteps, consume_less='mem', return_sequences=False)))

    elif M == "stacked_bidir":
    # test stacked with all RNN layers and consume_less options
        model = Sequential()
        model.add(embedding_layer)
        # model.add(Attention(recurrent.LSTM(embedding_dim, input_dim=embedding_dim,, consume_less='cpu' return_sequences=True))) # not supported
        model.add(Attention(recurrent.LSTM(output_dim=timesteps, consume_less='gpu', return_sequences=True)))
        model.add(Attention(recurrent.LSTM(output_dim=timesteps, consume_less='mem', return_sequences=True, go_backwards=True)))
        # test each other RNN type
        model.add(Attention(recurrent.GRU(output_dim=timesteps, consume_less='mem', return_sequences=True)))
        model.add(Attention(recurrent.SimpleRNN(output_dim=timesteps, consume_less='mem', return_sequences=False)))
        model.add(core.Activation('relu'))

    elif M == "simple_att":
        # test with return_sequence = False
        model = Sequential()
        model.add(embedding_layer)
        model.add(Attention(recurrent.LSTM(output_dim=timesteps, consume_less='mem',dropout_W=0.2, dropout_U=0.2)))
        model.add(core.Activation('relu'))

    elif M == "bidir_att":
    # with bidirectional encoder
        model = Sequential()
        model.add(embedding_layer)
        model.add(wrappers.Bidirectional(recurrent.LSTM(output_dim=timesteps, return_sequences=True)))
        model.add(Attention(recurrent.LSTM(output_dim=timesteps, return_sequences=False, consume_less='mem')))
        model.add(core.Activation('relu'))

    return model



# Building symbolic sentence models for [A] and [B] sides separately
def build_model(model_type, len_vocab_A, len_vocab_B, hidden_states, embedding_dim):

    sent_A=models(model_type, len_vocab_A, hidden_states, embedding_dim)#, DENSES)
    sent_B=models(model_type, len_vocab_B, hidden_states, embedding_dim)#, DENSES)

    if DENSES != 0:
        pair_sents=Merge([sent_A, sent_B], mode='concat', concat_axis=-1)
# -----------------------------------------------------------------------
        similarity = Sequential()
        similarity.add(pair_sents)
        similarity.add(MaxoutDense(DENSES))
        similarity.add(MaxoutDense(1))
    else:
        pair_sents=Merge([sent_A, sent_B], mode='dot', dot_axes=-1)
        similarity = Sequential()
        similarity.add(pair_sents)
        #similarity.add(MaxoutDense(1))
        similarity.add(core.Activation("sigmoid"))
    return similarity

import subprocess

def pearsons(gl, el):
    with open("GL.txt", "w") as f:
        for p in gl:
            f.write("%s\n" % p)
    with open("EL.txt", "w") as f:
        for p in el:
            f.write("%s\n" % p)
    gs="GL.txt"
    est="EL.txt"
    
    pipe = subprocess.Popen(["perl", "./correlation-noconfidence.pl", gs, est], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    pipe.stdin.write("%s %s" % (gs, est))
    try:
        pearson= float(str(pipe.stdout.read()).split()[1])
    except:
        print str(pipe.stdout.read())
        exit()

    remove(gs)
    remove(est)
    pipe.stdin.close()
    pipe.stdout.close()
    
    return pearson

if train:
    from pdb import set_trace as st

    similarity = build_model(MODEL_TYPE, len(word_index_A), 
                            len(word_index_B), h_STATES, EMBEDDING_DIM)

    print "Compiling the model..."
    similarity.compile(loss='mean_squared_error', optimizer='rmsprop', 
                    metrics=['acc','mean_squared_error'])

    similarity.get_config()
    summ=model_from_json(similarity.to_json(),custom_objects=dict(Attention=Attention))
    summ.summary()
    
    print "Happy learning!!!"
    checkpointer = ModelCheckpoint(filepath=model_file, monitor='val_acc', verbose=1, save_best_only=True)
    
    similarity.fit([x_train_A, x_train_B], y_train, validation_data=([x_val_A, x_val_B], y_val),
                                                 nb_epoch=EPOCHS, batch_size=20, callbacks=[checkpointer])

    print "\nParameters:\n---------------------\nh_STATES=%d\nEPOCHS=%d\nDENSES=%d\nEMBEDDING=%s\nEMBEDDING_DIM=%d\nMAX_SEQUENCE_LENGTH=%d\nMODEL_TYPE=%s\n" % (h_STATES,
                                                                                                                           EPOCHS,
                                                                                                                           DENSES,
                                                                                                                           EMBEDDING,
                                                                                                                           EMBEDDING_DIM,MAX_SEQUENCE_LENGTH,
                                                                                                                           MODEL_TYPE)
elif not train:
    import sys
    from os import remove
    from math import sqrt
    from sklearn.metrics import mean_squared_error as mse
    from sklearn.metrics import r2_score
    

    test_gs_file="gs_test_file"

    similarity=load_trained_model(model_file, MODEL_TYPE,len(word_index_A), 
                                   len(word_index_B), h_STATES, EMBEDDING_DIM)
    
    y=similarity.predict([x_data_Av, x_data_Bv])
    
    with open("/almac/ignacio/%s.pred" % params, "wb") as f:
        for score in  y[:,0]:
            f.write("%f.4\n" % score)

    with open(test_gs_file, "wb") as f:
        for score in test_labels:
            f.write("%f.4\n" % score)
# EVALUATING
    
    testRMSE = sqrt(mse(test_labels, y[:,0]))
    testR2 = r2_score(test_labels,y[:,0])
    testPea = pearsons(test_labels, y[:,0])
    
    sys.stderr.write('Test RMSE Score:     %.4f\n' % (testRMSE))
    sys.stderr.write('Test R2 Score:       %.4f\n' % (testR2))
    sys.stderr.write('Test wPearson Score: %.4f\n' % (testPea))
    
    remove(test_gs_file)
