import numpy as np
from sklearn.svm import SVR

inputfile = "/home/iarroyof/data/pairs_headlines_d2v_H300_sub_m5.mtx"
gsfile = "/home/iarroyof/data/STS.gs.headlines.txt"
outputfile = "/home/iarroyof/sem_outputs/svr_output_headlines_100_d2v_conc_300_m5.txt"

X = np.loadtxt(inputfile)
y = np.loadtxt(gsfile)
g = 8.3804

svr_rbf = SVR(kernel = 'rbf', C = 1e3, gamma = g).svr_rbf.fit(X, y)
y_rbf = svr_rbf.predict(X)

y={'estimated_output':y_rbf.tolist()}

with open(outputfile, "w") as f:
    f.write("%s" % y)
