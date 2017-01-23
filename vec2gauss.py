import matplotlib.pyplot as plt
from sklearn.datasets import make_gaussian_quantiles
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances as coss
from random import shuffle
from metric_learn import *

def gauss_k(x,x_i,s):
             return np.exp((np.sqrt((x-x_i).dot(x-x_i))**2)/2*s**2)

def gauss(X, width, weighted, indexes, g):
    H=[]; s=range(X.shape[0])
    mark=['.', ',', 'o', 'v', '^' ,  '<', '>', '1', '2', '3', '4', 's', 'p', '*', 'h', '+', 'x', '|', '_']
    R=np.array([X[j] for j in indexes])
    for x in R:
        if g:
            H.append([gauss_k(x, x_i, width, weighted) for x_i in X])
        else:
            H.append(x)
    plt.yscale('log'); j=0; s=range(len(H[0]))
    for S, i in zip(H,indexes):
        if not j % 2 == 0:
            lab = "H_"+str(i) + "  cos: %f.3" % coss(H[j], H[j-1])
        else:
            lab = "H_"+str(i)
        shuffle (mark)
        plt.plot(s, S, marker=mark[0], label=lab, linewidth=1.25); j += 1
    plt.legend(bbox_to_anchor=(1.05, 1), loc=1, borderaxespad=0.)
    plt.grid()
    plt.show()


dim=600
__op="conc"
width = 3.0

X1=np.loadtxt("/home/iarroyof/data/sts_all/pairs-NO/vectors_H%s/pairs_eng-NO-test-2e6-nonempty_d2v_H%s_%s_m5w8.mtx.half0" % (dim, dim, __op))
Y1=np.loadtxt("/home/iarroyof/data/sts_all/pairs-NO/STS.gs.all-eng-NO-test-nonempty-half0.txt")

X1, Y1 = make_gaussian_quantiles(n_features=2, n_classes=3)
X,Y = separate(X1,Y1)

gauss(X, width, False, [1745,1746,2099,3000, 9, 10, 451,452, 25,26,1579,1580])
