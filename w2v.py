# Author: Ignacio Arroyo-Fernandez (UNAM)

from gensim.models import Word2Vec, Doc2Vec
from gensim.models.doc2vec import TaggedLineDocument, LabeledSentence
import os
from argparse import ArgumentParser as ap
import sys

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
        sys.stderr.write('Badly formed string at the first attempt\n-- Document sample: '+gClean[0:49])
        try:
            range_error = 999
            normalized_string = normalize('NFKC', gClean[0:range_error].lower()) # One thousand of characters are written if available. 
        except TypeError:
            sys.stderr.write('\nThe wrong string at the second attempt: before %s words\n' % range_error)
            try:
                range_error = 99
                normalized_string = normalize('NFKC', gClean[0:range_error].lower())
            except TypeError:
                sys.stderr.write('\nThe wrong string at the third attempt: before %s words' % range_error)
                try:
                    range_error = 49
                    normalized_string = normalize('NFKC', gClean[0:range_error].lower())
                except TypeError:    
                    sys.stderr.write('\nIt was not possible forming output file after three attempts. Fatally bad file\n')
                    normalized_string = '# Fatally bad File\n'
                    pass
    return  normalized_string.split() # Return the unicode normalized document.

class yield_line_documents(object):
    def __init__(self, dirname, d2v=False):
        self.dirname = dirname
        self.d2v = d2v
    def __iter__(self):
        if self.d2v:
            for fname in os.listdir(self.dirname):
                l = 1
                for line in open(os.path.join(self.dirname, fname)):
                    yield LabeledSentence(clean_Ustring_fromU(line), str(l)+"_"+fname)
                    l += 1
        else:
            for fname in os.listdir(self.dirname):
                for line in open(os.path.join(self.dirname, fname)):
                    yield clean_Ustring_fromU(line)

if __name__ == "__main__":
    parser = ap(description='Trains and saves a word2vec model into a file for mmap\'ing. Tokenization is performed un utf-8 an for Python 2.7. Non-latin characters are replaced by spaces. The model is saved into a given directory. All options are needed.')    
    parser.add_argument('-i', type=str, dest = 'indir_file_name', help='Specifies the directory containing files to be processed. No sub-directories are allowed. In the case doc2vec is used, a file name must be specified. This file must contain a a sentence/document by line.')
    parser.add_argument('-o', type=str, dest = 'outfile', help='Specifies the file where to be stored the model.')
    parser.add_argument('-t', type=int, dest = 'threads', help='Specifies the number of threads the training will be divided.')
    parser.add_argument('-H', type=int, dest = 'hidden', help='Specifies the number of hidden units the model going to have.')
    parser.add_argument('-m', type=int, dest = 'minc', help='Specifies the minimum frequency a word should have in the corpus to be considered.')
    parser.add_argument('-d', default=False, action="store_true", dest = 'd2v', help='Toggles the doc2vec model, insted of the w2v one.')

    args = parser.parse_args()
    if args.d2v:
        #articles = TaggedLineDocument(args.indir_file_name)
        arts = yield_line_documents(args.indir_file_name, d2v = True)
        articles = []
        for a in arts:
            articles.append(a)
        d2v_model = Doc2Vec(articles, min_count = args.minc, workers = args.threads, size = args.hidden)    
        d2v_model.save(args.outfile, separately = None)
    else:
        articles = yield_line_documents(args.indir_file_name)
        w2v_model = Word2Vec(articles, min_count = args.minc, workers = args.threads, size = args.hidden)
        w2v_model.save(args.outfile, separately = None)
    
    model = Doc2Vec.load(args.outfile)
    del(model)
    sys.stderr.write("\n>> Finished !!\n")
        
