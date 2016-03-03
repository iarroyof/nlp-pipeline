# -*- coding: utf-8 -*-
import numpy as np
import sys
from scipy import signal
from gensim.models import Word2Vec as wtov

from pdb import set_trace as st

def T(x1, x2, name="convs"):
    """ Convolutional (or any other from the ones defined) operation between two CiS vectors 
    
    x1, x2 - vectors
    name - name of the operation (sub, conc, conv, corr), defaults to convolution
    """
    print "operating:", name
    if name.startswith("sub"):
        return x1 - x2
    if name.startswith("conc"):
        return np.concatenate((x1, x2))
    if name.startswith("convss"): # For short input vectors (dim < 500)
        return signal.convolve(x1, x2, mode="same")
    if name.startswith("convs"): # For large input vectors (dim > 500)
        return signal.fftconvolve(x1, x2, mode="same")
    if name.startswith("conv"): # For large input vectors and 2dim-1 output vectors
        return signal.fftconvolve(x1, x2)
    if name.startswith("corr"):
        return np.correlate(x1, x2, mode='same')

def clean_Ustring_fromU(string):
    from unicodedata import name, normalize
    gClean = ''
    for ch in u''.join(string.decode('utf-8', 'ignore')):
        try:
            if name(ch).startswith('LATIN') or name(ch) == 'SPACE':
                gClean = gClean + ch
            else: # Remove non-latin characters and change them by spaces
                gClean = gClean + ' '
        except ValueError: # In the case name of 'ch' does not exist in the unicode database.
            gClean = gClean + ' '
    
    try: # Trying different cases for bad input documents.
        normalized_string = normalize('NFKC', gClean.lower())
    except TypeError:
        sys.stderr.write('Bad formed string at the first attempt\n')
        try:
            range_error = 999
            normalized_string = normalize('NFKC', gClean[0:range_error].lower()) # One thousand of characters are written if available. 
        except TypeError:
            sys.stderr.write('\nThe wrong string at the second attempt: before %s words' % range_error)
            try:
                range_error = 99
                normalized_string = normalize('NFKC', gClean[0:range_error].lower())
            except TypeError:
                sys.stderr.write('\nThe wrong string at the third attempt: before %s words' % range_error)
                try:
                    range_error = 49
                    normalized_string = normalize('NFKC', gClean[0:range_error].lower())
                except TypeError:    
                    sys.stderr.write('\nIt was not possible forming output file after three attempts. Fatally bad file')
                    normalized_string = '# Fatally bad File\n'
                    pass
    return  normalized_string # Return the unicode normalized document.

def scsis(dws, sentence, log = False):
    """ Single Context summation inside a sentence
    
    Sum of all the word vectors in the sentence
    
    Recieves:
    dws - the db_word_space object to get the word vectors
    sentence - the sentence as a list of words
    log - Toggles logarithm of the output vectors    
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
    
def scsis_w2v(dws, sentence, log = False):
    """ Single Context summation inside a sentence
    
    Sum of all the word vectors in the sentence
    
    Recieves:
    dws - word2vec model
    sentence - as a list of words.
    log - Toggles logarithm of the output vectors
    """
    
    if not log:
        first = True
        while(first):
            try:
                word = sentence.pop(0)
                result = dws[word]
                first = False    
            except KeyError:
                sys.stderr.write(">> Word '"+word+"' is not in the model.\n")
                pass
            
        for word in sentence:
            try:
                result = result + dws[word]
            except KeyError:
                sys.stderr.write(">> Word '"+word+"' is not in the model.\n")
                pass
    else:
        first = True
        while(first):
            try:
                result = np.log2(dws[sentence.pop(0)])
                first = False    
            except KeyError:
                continue
            
        for word in sentence:
            try:
                result = result + np.log2(dws[word])
            except KeyError:
                continue                        

    return result                

def ccbsp(dws, s1, s2 = None, name = "convs", index = None):
    """ Context Convolution between Sentence Pairs
    
    Applies the T operation to the scsis vectors of the sentences
    
    Recieves:
    dws - the db_word_space object to get the word vectors
    s1, s2 - sentences to be combined, as lists of words
    name - name of the operation, defaults to convolution. If name ends with 'l', logarithm of the output vectors will be computed
    index - This is the 1-based index of the sentence pair (the row number in the input_file). It is needed only for doc2vec. The 
            reason is d2v uses tags for getting sentence vectors. Notice those vectors are not needed to be constructed, as in the 
            case of non-d2v (word-vector-based) representations. For sigle sentence by row, this index is not required.
    """
    if name.endswith('l'): # Toggles logarithm of the output
        log = True
    else:
        log = False
        
    if s2 != None:
        if 'word2vec' in str(dws.__class__):
            x1 = scsis_w2v(dws, s1)
            x2 = scsis_w2v(dws, s2)
        elif 'doc2vec' in str(dws.__class__):
            tag_x1 = str(index)+"_"+str(index+(2*index-3))+"_snippet" # the index i, the subindex: i + (2i-3)
            tag_x2 = str(index)+"_"+str(index+(2*index-2))+"_snippet" # the index i, the subindex: i + (2i-2)
            sys.stderr.write("\n%s\n" % dws.docvecs.doctags[tag_x1])
            sys.stderr.write("\n%s\n" % dws.docvecs.doctags[tag_x2])
            x1 = dws.docvecs[tag_x1]
            x2 = dws.docvecs[tag_x2]
        else:        
            x1 = scsis(dws, s1, log = log)
            x2 = scsis(dws, s2, log = log)
        
        return T(x1, x2, name)
    else:
        if 'doc2vec' in str(dws.__class__):
            tag = str(index)+"_sent"
            x = dws.docvecs[tag]
        elif 'word2vec' in str(dws.__class__):
            x = scsis_w2v(dws, s1, log = log)
        else:
            x = scsis(dws, s1, log = log)
            
        return x

def read_sentence_pairs(filename, n=False):
    """ Generator to read sentence pairs from "filename"
    
    Pairs must be separated by tab (\t). Yields a 3-tuple consisting of:
    (the pair number (the line in the file), the first sentence, the second sentence)
    The sentences are returned as list of words.
    If n is specified, it yields only n pairs.
    """
    with open(input_file) as fin:
        #row_i = 0
        for row_i, line in enumerate(fin):
            if not n is False and row_i == n:
                break
            ligne = line.split('\t')
            s1 = clean_Ustring_fromU(ligne[0]).split()
            s2 = clean_Ustring_fromU(ligne[1]).split()
            yield row_i, s1, s2    

def read_sentences(filename, n=False):
    """ Generator to read sentences from "filename"
    
    The sentences are returned as a list of words each.
    If n is specified, it yields only n sentences.
    """
    with open(input_file) as fin:
        for i, line in enumerate(fin):
            yield i, line.strip().lower().split()        
                        

if __name__ == "__main__":
    """Command line tool to read sentence pairs and generate combined output vectors file
    """
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", help="input file name (sentence pairs)", metavar="input_file", required=True)
    parser.add_argument("-d", help="name of the database (word space). Type '-d word2vec' instead if you are using Word2Vec. The default file is 'word2vec.model' in current directory. Also '-d doc2vec' can be specified. For both latter options, use -w option for specifying the model file.", metavar="database", required=True)
    parser.add_argument("-o", help="output file name (optional, defaults to output.mtx)", default="output.mtx", metavar="output_file")
    parser.add_argument("-l", help="limit to certain number of pairs (optional, defaults to whole file)", default=False, metavar="limit")
    parser.add_argument("-S", help="Toggles generation of single sentence vectors (not pairs). The input file must contain a sentence or text by line.", action="store_true")    
    parser.add_argument("-t", help="combiner operation, can be {conc: Simple_concatenation, corr:cross_correlation_same_output_dimension, conv:large_input_convolution (dim > 500), convs:large_input_convolution_same_output_dimension, convss:short_input_convolution_same_output_dimension (dim < 500), conc:concatenation, sub:subtraction}. Place an 'l' at the end of any operation for getting output logarithmic transformation of your vectors.", metavar="operation") 
    parser.add_argument("-s", help="save as sparse", action="store_true")
    parser.add_argument("-F", help="frequency threshold", metavar="threshold")
    parser.add_argument("-w", help="Word2vec model allocation. Default=word2vec.model", default="word2vec.model", metavar="word2vec_model")
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
        if args.F:
            dws = d_ws.db_word_space(args.d, args.F)
        else:
            dws = d_ws.db_word_space(args.d)        
    else:
        dws = wtov.load(args.w)
    
    if args.s and args.d != 'word2vec' and not args.S:
        if not args.t: 
            print '-t argument is required.'
            exit()
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
    elif not args.s and not args.S:
        if not args.t: 
            print '-t argument is required.'
            exit()
        import os
        from numpy import savetxt
        if os.path.isfile(output_file):
            os.unlink(output_file)
        invalid_rows = "invalid.txt"
        open(invalid_rows, "w").close()
        with open(output_file, "a") as fout:
            
            for j, s1, s2 in read_sentence_pairs(input_file, limit):
                try:                                         # 1-based indexing
                    v = np.array(ccbsp(dws, s1, s2, operation, j+1)).astype("float64") 
                except IndexError:
                    with open(invalid_rows, 'a') as finv:
                        finv.write(str(j+1)+"\n") # 1-based indexing
                    sys.stderr.write("A sentence is completely absent (w2v/db) -[%s]- %s -- %s\n" % (j+1, s1, s2))
                except KeyError:
                    with open(invalid_rows, 'a') as finv:
                        finv.write(str(j+1)+"\n")
                    sys.stderr.write("A sentence is completely absent (d2v) -[%s]- %s -- %s\n" % (j+1, s1, s2))
                savetxt(fout, v, newline=' ')
                fout.write("\n"); 
    elif args.S:
        import os
        from numpy import savetxt
        if os.path.isfile(output_file):
            os.unlink(output_file)
        with open(output_file, "a") as fout:
            for j, s in read_sentences(input_file, limit):
                try:
                    v = np.array(list(ccbsp(dws = dws, s1 = s, index = j+1))).astype('float64')
                except KeyError:
                    sys.stderr.write("A sentence or key is completely absent (d2v) -[%s]-- %s\n" % (j+1, s))
                except IndexError:
                    sys.stderr.write("A sentence is completely absent (w2v/db) -[%s]-- %s\n" % (j+1, s))
                savetxt(fout, v, newline=' ')
                fout.write("\n")    
    
