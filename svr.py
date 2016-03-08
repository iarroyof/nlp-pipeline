import numpy as np
from sklearn.svm import SVR
from sklearn.grid_search import RandomizedSearchCV as RS
from scipy.stats import expon, uniform

#inputfile = "/home/ignacio/data/vectors/pairs_headlines_d2v_H300_sub_m5.mtx"
#inputfile = "/home/iarroyof/data/pairs_headlines_d2v_H300_sub_m5.mtx"
inputfile = "/home/iarroyof/data/pairs_headlines_d2v_H300_convss_m5.mtx"
gsfile = "/home/iarroyof/data/STS.gs.headlines.txt"
#outputfile = "/home/iarroyof/sem_outputs/svr_output_headlines_100_d2v_conc_300_m5.txt"
N = 5
X = np.loadtxt(inputfile)
y = np.loadtxt(gsfile)
op = 'convs'

gammas = {
        'conc': expon(scale=10, loc=8.38049430369), 
        'sub': expon(scale = 10, loc=15.1454004504), 
        'convs':expon(scale = 50, loc = 541.113519625)}

param_grid = [   
    {'C': [1, 10, 100, 1000, 1500, 2000], 'kernel': ['poly'], 'degree':[1, 2, 3, 4, 5]},
    {'C': [1, 10, 100, 1000, 1500, 2000], 'gamma': gammas[op], 'kernel': ['rbf']} ]

for n in xrange(N):
    for params in param_grid:
        svr = SVR()
        rs = RS(svr, param_distributions = params, n_iter = 10, n_jobs = 4)
        rs.fit(X, y)
        y_est = rs.predict(X).tolist()

        y_out = {}
        y_out['estimated_output'] = y_est
        y_out['best_params'] = rs.best_params_
        y_out['best_score'] = rs.best_score_

        with open("svr_output_headlines_100_d2v_%s_300_m5.txt" % (op), "a") as f:
            f.write(str(y_out)+'\n')

