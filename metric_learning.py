"""
This a Python script for sequential tests of all Metric Learning algorithms 
provided by metric-learn module (Metric Learning in Python). 

http://all-umass.github.io/metric-learn

This script is prepared for a custom collection from the SemEval STS whole 
dataset.

Ignacio Arroyo-Fernandez
IIMAS -- UNAM
"""
from numpy import zeros, array
import numpy as np
from data_theano import *

dim=300
__op="conc"

def separate(X_c, Y_c):
    (r, c) = X_c.shape
    Y = zeros((2 * r, ))
    X = zeros((2 * r, c/2))
    for i in xrange(r):
        X[2*i], X[2*i+1] = X_c[i, 0:c/2], X_c[i, c/2:c]
        Y[2*i], Y[2*i+1] = Y_c[i], Y_c[i]

    return X, Y

def predict(X, M, Y, pairs=False):
    """
    math: for each $x,y \in X$ do: $D(x,y)=\sqrt{(x-y)^T M(x-y)}$
    """
    
    from sklearn.metrics import r2_score as R2
    from scipy.stats import pearsonr

    # Theano implementation
    #import theano
    #import theano.tensor as T

    #D = T.matrix('D')
    #S = T.vector('S')
    #Z = T.vector('Z')
    #dots = T.dot(S - Z, T.dot(D, S - Z))
    #sqrd_metric = theano.function(inputs=[S, D, Z], 
    #                              outputs=scan(lambda : d, T.sqrt(d), 
    #                              sequences=dots)
    #                              )
    #y_pred = sqrd_metric(X, M, X)
    if not pairs:
        odds = range(len(X))[1::2]
        pair = range(len(X))[::2]

        y_pred = [(X[a]-X[b]).dot(M).dot(X[a]-X[b])
                                for a, b in zip(pair, odds)]
    # For pretransformed vectors                  
    #y_pred = array([np.sqrt((X[a]-X[b]).dot(X[a]-X[b])) for a, b in zip(pair, odds)])
    else:
        y_pred = [np.sqrt(x.dot(M.T).dot(x.dot(M.T)) ) for x in X]

    return (#y_pred, Y, #R2(Y, y_pred, multioutput='variance_weighted'), 
            pearsonr(Y, y_pred)[0])
    
if __name__ == '__main__':
    from metric_learn import Covariance, ITML_Supervised, SDML_Supervised, LSML_Supervised
    
#    metrics = [Covariance, ITML_Supervised, SDML_Supervised, LSML_Supervised]
    metrics = [SDML_Supervised]

    path = "/almac/ignacio/data/sts_all/"
    tr_px = path + "pairs-SI/vectors_H%s/pairs_eng-SI-test-2e6-nonempty_d2v_H%s_%s_m5w8.mtx" % (dim, dim, __op)
    tr_py = path + "pairs-SI/STS.gs.all-eng-SI-test-nonempty.txt"
    ts_px = path + "pairs-NO/vectors_H%s/pairs_eng-NO-test-2e6-nonempty_d2v_H%s_%s_m5w8.mtx.half0" % (dim, dim, __op)
    ts_py = path + "pairs-NO/STS.gs.all-eng-NO-test-nonempty-half0.txt"
    vl_px = path + "pairs-NO/vectors_H%s/pairs_eng-NO-test-2e6-nonempty_d2v_H%s_%s_m5w8.mtx.half1" % (dim, dim, __op)
    vl_py = path + "pairs-NO/STS.gs.all-eng-NO-test-nonempty-half1.txt"

    datasets = load_my_data(tr_px,tr_py,ts_px,ts_py,vl_px,vl_py, shared=False)
    #valid_set_x, valid_set_y = datasets[1]

# Load training and test sets
    tr_set_x, tr_set_y = datasets[0]
    # For independents
    #X, Y = separate(tr_set_x, tr_set_y) 

    ts_set_x, ts_set_y = datasets[2]
    #x, y = separate(ts_set_x, ts_set_y)

# Fit metric under the suposition that all those training vectors sharing label are closer.
# TODO: Add ifs for giving parameters for each metric learning algorithm.
    for metric in metrics:
        print ">>>> ", metric
        m = metric(num_constraints=200, balance_param=0.7, sparsity_param=1.9, use_cov=True, verbose=True)
        m.fit(tr_set_x, tr_set_y)
  
        M = m.metric()
        X_ = m.transform(tr_set_x)
        L = m.transformer()

       # print predict(ts_set_x, L, ts_set_y, pairs=True)
        print predict(ts_set_x, L.T.dot(L), ts_set_y, pairs=True) #<<<<<<<<<<<< For pairs
       # Max\Rho = 0.274591637688
      #  print predict(x, M.T.dot(M), ts_set_y)
     #   print predict(x, M, ts_set_y)

    #    print predict(x, np.linalg.inv(M.T.dot(M)), ts_set_y)
   #     print predict(x, np.linalg.inv(M), ts_set_y)

        print predict(ts_set_x, X_.T.dot(X_), ts_set_y, pairs=True) # <<<<<<< For independents
        # Max\Rho = -0.186109504445
        #print predict(x, np.linalg.inv(X_.T.dot(X_)), ts_set_y)
  
 #       print predict(X_, np.identity(len(X_[0])), ts_set_y)
