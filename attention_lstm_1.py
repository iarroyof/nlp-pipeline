"""
A keras attention layer that wraps RNN layers.

Based on tensorflows [attention_decoder](https://github.com/tensorflow/tensorflow/blob/c8a45a8e236776bed1d14fd71f3b6755bd63cc58/tensorflow/python/ops/seq2seq.py#L506) 
and [Grammar as a Foreign Language](https://arxiv.org/abs/1412.7449).

date: 20161101
author: wassname
url: https://gist.github.com/wassname/5292f95000e409e239b9dc973295327a
"""

# test likes in https://github.com/fchollet/keras/blob/master/tests/keras/layers/test_wrappers.py
import pytest
import numpy as np
from numpy.testing import assert_allclose
from keras.utils.test_utils import keras_test
from keras.layers import wrappers, Input, recurrent, InputLayer, Merge,MaxoutDense
from keras.layers import core, convolutional, recurrent, Embedding, Dense
from keras.models import Sequential, Model, model_from_json
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical

from attention_lstm_ import *
from load_sts_data import *

YEAR_TRAIN="2013"
YEAR_VALID="2017"
MAX_SEQUENCE_LENGTH=50
VALIDATION_SPLIT=0.30
#representation = "fastText"
#representation = "word2vec"
representation = "glove"
VECTOR_DIR="/almac/ignacio/data/" + representation
EMBEDDING_DIM=100
TRAIN_DIRS=[(VECTOR_DIR.rsplit('/', 1)[0]
 + "/sts_all/train-" + YEAR_TRAIN, None, False)]

VALID_DIRS=[(VECTOR_DIR.rsplit('/', 1)[0]
 + "/sts_all/valid-" + YEAR_VALID, "validation", False)]
MAX_NB_WORDS=20000

# --------------------------
print "Loanding train and valid dirs......"
train_data_, gs_data=load_train_dirs(TRAIN_DIRS)
valid_data_, _ =load_train_dirs(VALID_DIRS)

print "Spliting tab-separated files..."
train_data_A, train_data_B = train_data_[1::2], train_data_[::2]
valid_data_A, valid_data_B = valid_data_[1::2], valid_data_[::2]

#labels = to_categorical(np.asarray(gs_data))
labels=np.asarray(gs_data)
indices = np.arange(labels.shape[0])
np.random.shuffle(indices)
nb_validation_samples = int(VALIDATION_SPLIT * labels.shape[0])

print "Labels shape: ", labels.shape

print "Tokenizing files... [A]"
tokenizer = Tokenizer(nb_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(train_data_A + valid_data_A)
sequences_A = tokenizer.texts_to_sequences(train_data_A)
sequences_Av = tokenizer.texts_to_sequences(valid_data_A)
word_index_A = tokenizer.word_index
data_A = pad_sequences(sequences_A, maxlen=MAX_SEQUENCE_LENGTH)
x_data_Av = pad_sequences(sequences_Av, maxlen=MAX_SEQUENCE_LENGTH)
data_A = data_A[indices]

print "Split training set into train and val... [A]"
x_train_A = data_A[:-nb_validation_samples]
x_val_A = data_A[-nb_validation_samples:]

print "Tokenizing files... [B]"
tokenizer = Tokenizer(nb_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(train_data_B + valid_data_B)
sequences_B = tokenizer.texts_to_sequences(train_data_B)
sequences_Bv = tokenizer.texts_to_sequences(valid_data_B)
word_index_B = tokenizer.word_index
data_B = pad_sequences(sequences_B, maxlen=MAX_SEQUENCE_LENGTH)
x_data_Bv = pad_sequences(sequences_Bv, maxlen=MAX_SEQUENCE_LENGTH)
data_B = data_B[indices]

print "Split training set into train and val... [B]"
x_train_B = data_B[:-nb_validation_samples]
x_val_B = data_B[-nb_validation_samples:]

labels = labels[indices]
y_train = labels[:-nb_validation_samples]
y_val = labels[-nb_validation_samples:]

embeddings_index = {}
if representation == "glove":
    f = open(os.path.join(VECTOR_DIR, 'glove.6B.%dd.txt' % EMBEDDING_DIM))
elif representation == "fastText":
    f = open(os.path.join(VECTOR_DIR, 'wikiEn_Full_H%d.model.vec' % EMBEDDING_DIM))
elif representation == "word2vec":
    f = open(os.path.join(VECTOR_DIR, 'w2v_En_vector_space_H%d.vec' % EMBEDDING_DIM))

print "Getting embedding matrix..."
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

# Leaning constants
outfile="probabilities_bidir"
h_STATES = 25
EPOCHS = 100
DENSES = 10
MODEL_TYPE="stacked"
timesteps=h_STATES
embedding_dim=EMBEDDING_DIM

# Building symbolic sentence models for [A] and [B] sides separately
sent_A=models(MODEL_TYPE, len(word_index_A), timesteps, embedding_dim)#, DENSES)
sent_B=models(MODEL_TYPE, len(word_index_B), timesteps, embedding_dim)#, DENSES)

pair_sents=Merge([sent_A, sent_B], mode='concat', concat_axis=-1)
# -----------------------------------------------------------------------

similarity = Sequential()
similarity.add(pair_sents)
similarity.add(MaxoutDense(DENSES))
similarity.add(MaxoutDense(1))
#similarity.add(Dense(5, activation="softmax"))
#similarity.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
similarity.compile(loss='mean_absolute_error', optimizer='rmsprop', metrics=['mean_absolute_error','mean_squared_error'])

print similarity.summary()

# happy learning!
similarity.fit([x_train_A, x_train_B], y_train, validation_data=([x_val_A, x_val_B], y_val),
          nb_epoch=EPOCHS, batch_size=20)

print "\nParameters:\n---------------------\nh_STATES=%d\nEPOCHS=%d\nDENSES=%d\nRepresentation=%s\nEMBEDDING_DIM=%d\nMAX_SEQUENCE_LENGTH=%d\nMODEL_TYPE=%s\n" % (h_STATES,
                                                                                                                           EPOCHS,
                                                                                                                           DENSES,
                                                                                                                           representation,
                                                                                                                           EMBEDDING_DIM,MAX_SEQUENCE_LENGTH,
                                                                                                                           MODEL_TYPE)
