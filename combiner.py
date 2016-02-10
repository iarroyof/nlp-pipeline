# -*- coding: utf-8 -*-
import numpy as np
import sys
from scipy import signal
from gensim.models import Word2Vec as wtov

def T(x1, x2, name="convs"):
    """ Convolutional (or any other from the ones defined) operation between two CiS vectors 
    
    x1, x2 - vectors
    name - name of the operation (sub, conc, conv, corr), defaults to convolution
    """
    if name.startswith("sub"):
        return x1 - x2
    if name.startswith("conc"):
        return np.concatenate((x1, x2))
    if name.startswith("convs"): # For large input vectors (dim > 500)
        return signal.fftconvolve(x1, x2, mode="same")
    if name.startswith("convss"): # For short input vectors (dim < 500)
        return signal.convolve(x1, x2, mode="same")
    if name.startswith("conv"): # For large input vectors and 2dim-1 output vectors
        return signal.fftconvolve(x1, x2)
    if name.startswith("corr"):
        return np.correlate(x1, x2, mode='same')

def scsis(dws, sentence, log = False):
    """ Single Context summation inside a sentence
    
    Sum of all the word vectors in the sentence
    
    Recieves:
    dws - the db_word_space object to get the word vectors
    sentence - the sentence as a list of words
    w2v - Toggles word2vec using
    """
    if log:
        result = np.log2(dws.word_vector(sentence.pop(0)) + 1)
        for word in sentence:
            result = result + np.log2(dws.word_vector(word) + 1)
    else:
        result = dws.word_vector(sentence.pop(0))
        for word in sentence:
            result = result + dws.word_vector(word)

    return result            
    
def scsis_w2v(dws, sentence):
    """ Single Context summation inside a sentence
    
    Sum of all the word vectors in the sentence
    
    Recieves:
    dws - the db_word_space object to get the word vectors
    sentence - the sentence as a list of words
    w2v - Toggles word2vec using
    """

    first = True
    while(first):
        try:
            result = dws[sentence.pop(0)]
            first = False    
        except KeyError:
            continue
            
    for word in sentence:
        try:
            result = result + dws[word]
        except KeyError:
            continue

    return result                

def ccbsp(dws, s1, s2, name="convs"):#, w2v = False):
    """ Context Convolution between Sentence Pairs
    
    Applies the T operation to the scsis vectors of the sentences
    
    Recieves:
    dws - the db_word_space object to get the word vectors
    s1, s2 - sentences to be combined, as lists of words
    name - name of the operation, defaults to convolution
    """
    if 'word2vec' not in str(dws.__class__):
        x1 = scsis(dws, s1)
        x2 = scsis(dws, s2)
    else:
        x1 = scsis_w2v(dws, s1)
        x2 = scsis_w2v(dws, s2)
        
    return T(x1, x2, name)

def read_sentence_pairs(filename, n=False):
    """ Generator to read sentence pairs from "filename"
    
    Pairs must be separated by tab (\t). Yields a 3-tuple consisting of:
    (the pair number (the line in the file), the first sentence, the second sentence)
    The sentences are returned as list of words.
    If n is specified, it yields only n pairs.
    """
    with open(input_file) as fin:
        for row_i,line in enumerate(fin):
            if not n is False and row_i == n:
                break
            s1, s2 = line.strip().split("\t")
            s1 = s1.lower().split()
            s2 = s2.lower().split()
            yield row_i, s1, s2    

if __name__ == "__main__":
    """Command line tool to read sentence pairs and generate combined output vectors file
    """
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", help="input file name (sentence pairs)", metavar="input_file", required=True)
    parser.add_argument("-d", help="name of the database (word space). Type '-d word2vec' instead if you are using Word2Vec. This option asumes your word2vec model is in the current work directory and it is called word2vec.model", metavar="database", required=True)
    parser.add_argument("-o", help="output file name (optional, defaults to output.mtx)", default="output.mtx", metavar="output_file")
    parser.add_argument("-l", help="limit to certain number of pairs (optional, defaults to whole file)", default=False, metavar="limit")
    parser.add_argument("-t", help="combiner operation, can be {corr:cross_correlation_same_output_dimension, conv:large_input_convolution (dim > 500), convs:large_input_convolution_same_output_dimension, convss:short_input_convolution_same_output_dimension (dim < 500), conc:concatenation, sub:subtraction}.", metavar="operation", required=True) 
    parser.add_argument("-s", help="save as sparse", action="store_true")
    args = parser.parse_args()
    
    input_file = args.f
    output_file = args.o
    limit = int(args.l) or False
    operation = args.t
    row = []
    col = []
    data = []
    #For sparse data a csr matrix is constructed from the coordinates
    
    if args.d != 'word2vec':
        import db_word_space as d_ws
        dws = d_ws.db_word_space(args.d)
    else:
        dws = wtov.load('word2vec.model')
        
    if args.s and args.d != 'word2vec':
        for row_i, s1, s2 in read_sentence_pairs(input_file, limit):
            v = ccbsp(dws, s1, s2, operation)
            for col_i in range(0,len(v)):
                if v[col_i]:
                    row.append(row_i)
                    col.append(col_i)
                    data.append(v[col_i])
        from scipy.sparse import csr_matrix
        from scipy import io
        m = csr_matrix((data, (row, col)))
        io.mmwrite(output_file, m)    
    #Otherwise the vectors are appended row-wise to a file
    else:
        import os
        from numpy import savetxt
        if os.path.isfile(output_file):
            os.unlink(output_file)
        with open(output_file, "a") as fout:
            for row_i, s1, s2 in read_sentence_pairs(input_file, limit):
                v = ccbsp(dws, s1, s2, operation).astype("float32")
                savetxt(fout, v, newline=' ')
                fout.write("\n")
                
