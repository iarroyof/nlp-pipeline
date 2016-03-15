import numpy as np
from sklearn.svm import SVR
from sklearn.grid_search import RandomizedSearchCV as RS
from scipy.stats import randint as sp_randint
from scipy.stats import expon
from sklearn.externals import joblib

#inputfile = "/home/ignacio/data/vectors/pairs_headlines_d2v_H300_sub_m5.mtx"
#inputfile = "/home/iarroyof/data/pairs_headlines_d2v_H300_sub_m5.mtx"
corpus = "puces_complete"
representation = "d2v"
dimensions = "H300"
min_count = "m10"

from argparse import ArgumentParser as ap
parser = ap(description='This script trains a SVR over any input dataset of numerical representations. The main aim is to determine a set of learning parameters')
parser.add_argument("-x", help="Input file name (vectors)", metavar="input_file", required=True)
parser.add_argument("-y", help="Regression  labels file.", metavar="regrLabs_file", required=True)
parser.add_argument("-n", help="Number of tests to be performed.", metavar="tests_amount", default=1)
parser.add_argument("-o", help="The operation the input data was derived from. Options: {'conc', 'convs', 'sub'}. In the case you want to give a precalculated center for width randomization, specify the number. e.g. '-o 123.654'.", default="conc")
args = parser.parse_args()
#inputfile = "/home/iarroyof/data/pairs_headlines_d2v_H300_convss_m5.mtx"
#gsfile = "/home/iarroyof/data/STS.gs.headlines.txt"
#outputfile = "/home/iarroyof/sem_outputs/svr_output_headlines_100_d2v_conc_300_m5.txt"
N = int(args.n)
X = np.loadtxt(args.x)
y = np.loadtxt(args.y)
op = args.o

if op.replace('.','',1).isdigit():
    op = 'esp'

gammas = {
        'conc': expon(scale=10, loc=8.38049430369), 
        'sub': expon(scale = 20, loc=15.1454004504), 
        'convs':expon(scale = 50, loc = 541.113519625),
        'esp':expon(scale = 20, loc = float(args.o)) }

param_grid = [   
    {'C': [1, 10, 100, 1000, 1500, 2000], 'kernel': ['poly'], 'degree': sp_randint(1, 5)},
    {'C': [1, 10, 100, 1000, 1500, 2000], 'gamma': gammas[op], 'kernel': ['rbf']} ]

for n in xrange(N):
    for params in param_grid:
        svr = SVR()
        rs = RS(svr, param_distributions = params, n_iter = 10, n_jobs = 24)
        rs.fit(X, y)
        f_x = rs.predict(X).tolist()
        num_lines = sum(1 for line in open("svr_%s_%s_%s_%s_%s.out" % (corpus, representation, dimensions, op, min_count), "r"))        

        y_out = {}
        y_out['estimated_output'] = f_x
        y_out['best_params'] = rs.best_params_
        y_out['learned_model'] = {'file': "pkl/svr_%s_%s_%s_%s_%s_%s.model" % (num_lines, corpus, representation, dimensions, op, min_count) }
        y_out['performance'] = rs.best_score_

        with open("svr_%s_%s_%s_%s_%s.out" % (corpus, representation, dimensions, op, min_count), "a") as f:
            f.write(str(y_out)+'\n')
        
        joblib.dump(svr, "pkl/svr_%s_%s_%s_%s_%s_%s.model" % (num_lines, corpus, representation, dimensions, op, min_count)) 
