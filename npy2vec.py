import numpy as np
# Program for converting npz word vectors from
# http://vsmlib.readthedocs.io/en/latest/tutorial/getting_vectors.html
# into text word2vec format

matrix="words500.npy"
vocab_file="words500.vocab"
# words500.npy  words500.vocab
class BeyonceIterable(object):
    def __iter__(self):
        for word in open(vocab_file, "r"):
            yield word

vocab=BeyonceIterable()

X=np.load(matrix, mmap_mode='r')

with open(matrix.split('.')[0] + ".vec", "w") as f:
    f.write("%d %d\n" % X.shape)
    for word, v in zip(vocab, X):
        f.write("%s %s\n" % (word.strip(), np.array2string(v,
                                                formatter={'float_kind':lambda x: "%.6f" % x}, 
                                                  max_line_width=20000).strip(']').strip('[') ))
