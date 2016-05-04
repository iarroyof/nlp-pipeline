import numpy as np
from sklearn.svm import SVR, NuSVR
from sklearn.grid_search import RandomizedSearchCV as RS
from scipy.stats import randint as sp_randint
from scipy.stats import expon
from sklearn.externals import joblib
from sklearn.metrics import r2_score
from re import search, M, I
from load_regression import load_regression_data as lr
import sys

def tener(x):
    return x * 10
def detener(x):
    return x / 10

from argparse import ArgumentParser as ap
parser = ap(description='This script trains/applies a SVR over any input dataset of numerical representations. The main aim is to determine a set of learning parameters')
parser.add_argument("-x", help="Input file name (vectors)", metavar="input_file", required=True)
parser.add_argument("-y", help="""Regression labels file. Do not specify this argument if you want to uniauely predict over any test set. In this case, you must to specify
                                the SVR model to be loaded as the parameter of the option -o.""", metavar="regrLabs_file", default = None)
parser.add_argument("-n", help="Number of tests to be performed.", metavar="tests_amount", default=1)
parser.add_argument("-o", help="""The operation the input data was derived from. Options: {'conc', 'convss', 'sub'}. In the case you want to give a precalculated center for
                                width randomization, specify the number. e.g. '-o 123.654'. A filename can be specified, which is the file where a SVR model is sotred,
                                e.g. '-o filename.model'""", metavar="operat{or,ion}")
parser.add_argument("-c", help="Sentence corpus over wich the learning will be performed, e.g. '-c headlines13' which indicates headlines (year 2013) corpus.", 
                    metavar="corpus_name", default = None)
parser.add_argument("-r", help="The representation type. options:={w2v, d2v, coocc, doc, docLog, docLogSvd, coocPMI, cooccLogSVD, cooccPMIsvd}.", 
                    metavar="repr_name", default = None)
parser.add_argument("-d", help="Dimensions of the input vector set.", metavar="num_dimensions", default = None)
parser.add_argument("-u", help="Especify C regulazation parameter. For a list '-u C:a_b', for a value '-u C:a'.", metavar="fixed_params", default = None)
parser.add_argument("-m", help="Minimun word frequency taken into account in the corpus. Type '-m m10' for a min_count of 10.", metavar="min_count", default = None)
parser.add_argument("-p", help="Toggle if you will process pairs.", action="store_true", default = False)
parser.add_argument("-N", help="""Toggle if NuSVR (valid only for NuSVR) will be used. The portion of suppoort vectors with respect to the training set 
                                  can be specified in [0,1]. If it is not specified, random values will be generated (normally 
                                  distributed with mean 0.35).""", metavar = "Nu", default = None)
parser.add_argument("-K", help="Kernel type custom specification. Uniquely valid if -u is not none.  Options: gaussian, linear, sigmoid.", metavar="kernel", default = None)
parser.add_argument("-s", help="Toggle if you will process sparse input format.", action="store_true", default = False)
parser.add_argument("-t", help="Toggle if you want multiply by 10 the regression labels for training.", action="store_true", default = False)
parser.add_argument("-k", help="k-fold cross validation for the randomized search.", metavar="k-fold_cv", default=None)
args = parser.parse_args()

N = int(args.n)

try:
    source = search(r"(?:vectors|pairs)_([A-Za-z\-]+[0-9]{0,4})_?(T[0-9]{2,3}_C[1-9]_[0-9]{2})?_([d2v|w2v|coocc\w*|doc\w*]*)_(H[0-9]{1,4})_?([sub|co[nvs{0,2}|rr|nc]+]?)?_(m[0-9]{1,3}[_[0-9]{0,3}]?)", args.x, M|I)
# example filename: 'pairs_headlines13_T01.._d2v_H300_conc_m5.mtx'
    if args.c:     #           1        2*    3    4   5*  6  
        corpus = args.c
    else:
        corpus = source.group(1)
    if args.r:
        representation = args.r
    else:
        representation = source.group(3)
    if args.d:
        dimensions = args.d
    else:
        dimensions = source.group(4)[1:]
    if args.m:
        min_count = args.m
    else:
        min_count = source.group(6)[1:]
except IndexError:
    print "\nError in the filename. One or more indicators are missing. Notation: <vectors|pairs>_<source_corpus>_<model_representation>_<dimendions>_<operation>*_<minimum_count>.mtx\n"
    for i in range(6):
        try:
            print source.group(i)
        except IndexError:
            print ":>> Unparsed: %s" % (i)
            pass
    exit()
except AttributeError:
    print "\nFatal Error in the filename. Notation: <vectors|pairs>_<source_corpus>_<model_representation>_<dimendions>_<operation>*_<mminimum_count>.mtx\n"
    for i in range(6):
        try:
            print source.group(i)
        except AttributeError:
            print ":>> Unparsed: %s" % (i)
            pass            
    exit()

print "\nCorpus: %s\nRepr: %s\nDimms: %s\nF_min: %s\nOpperation: %s\n" % (corpus, representation, dimensions, min_count, source.group(5))

if args.s:  # fileTrain = None, fileTest = None, fileLabelsTr = None, fileLabelsTs = None, sparse=False
            # features_tr, features_ts, labels_tr, labels_ts
   X, Xt, y, yt  = lr(fileTrain = args.x, fileLabelsTr = args.y, sparse = args.s)
else:
    X = np.loadtxt(args.x)
    if args.y:
        y = np.loadtxt(args.y)

if args.N != None:
    svr_ = "nu_svr"
else:
    svr_ = "svr"

gammas = {
        'conc': expon(scale=10, loc=8.38049430369),
        'sub': expon(scale = 20, loc=15.1454004504),
        'convss':expon(scale = 50, loc = 541.113519625),
        'corr':expon(scale = 50, loc = 631.770) }

if args.o:
    if args.o.replace('.','',1).isdigit():
        op = 'esp'
        gammas[op] = expon(scale = 20, loc = float(args.o))

    elif not args.o in gammas: # then it is a SVR model location
        from os.path import basename, splitext 
        # example filename: 'pairs_headlines13_d2v_H300_conc_m5.mtx'
        op = args.o

        sys.stderr.write("\n:>> Source: %s\n" % (source.group()))
        infile = basename(op) # SVR model file name
        model_idx = search(r"_(\d{1,3})_[d|w]2v_H",infile, I|M).group(1)
        if infile and infile != "*": # svr_output_headlines100_d2v_convs_300_m5.txt      
            filename = "svr_idx%s_%s_%s_H%s_predictions.out" % (model_idx, corpus, representation, dimensions)
            model = joblib.load(op, 'r')
            y_out = {}
            if args.t:
                y_out['estimated_output'] = map(detener, model.predict(X).tolist()) # Get back rescalling
            else:
                y_out['estimated_output'] = model.predict(X).tolist()
            if source.group(2):
                y_out['source'] = source.group(2)+".txt"
            else:
                y_out['source'] = corpus
            y_out['model'] = op
        # Add more metadata to the dictionary as being required.
            with open(filename, 'a') as f:
                f.write(str(y_out)+'\n')
            sys.stderr.write("\n:>> Output predictions: %s\n" % (filename))
            exit()
        else:
            print "Please specify a file name for loading the SVR pretrained model."            
            exit()
else:
    # example filename: 'pairs_headlines13_T01.._d2v_H300_conc_m5.mtx'
    if source.group(5):#          1         2    3    4    5   6 
        op = source.group(5)
    else:
        print "Please specify an operation type or give it in the correct way at the filename."
        exit()
        
sys.stderr.write("\n:>> Source: %s\n" % (source.group(1)))

if args.u:
    kd = {'Cr': search(r"C:([+-]?\d+(\.\d+)?)_(\d+(\.\d+)?)", args.u, I|M), 'Cu': search(r"C:([-+]?([^_aA-zZ]*$))", args.u, I|M) }
    if args.K:
        kernel = args.K
    else:
        kernel = 'rbf'

    if kd['Cr'] and not kd['Cu']: # For a specified range of C's
        a = float(kd['Cr'].group(1))
        b = float(kd['Cr'].group(3))
        assert 0 < a <= b and args.o.replace('.','',1).isdigit() # Badly specified range or gamma
        param_grid = [{'C': list(np.random.uniform(low=a, high=b, size=(10,))), 'gamma': [float(args.o)], 'kernel': [kernel]}]
    elif kd['Cu'] and not kd['Cr']: # For an unique C
        assert 0 < float(kd['Cu'].group(1)) and args.o.replace('.','',1).isdigit() # Badly specified range or gamma
        #gamms = [float(args.o)-float(args.o)*0.2, float(args.o), float(args.o)+float(args.o)*0.2]
        #gamms = [float(args.o)]
        gamms = expon(scale=2, loc=float(float(args.o)))
        #param_grid = [{'C': [float(kd['Cu'].group(1))], 'gamma': gamms, 'kernel': [kernel]}]
        param_grid = [{'C': expon(scale=2, loc=float(kd['Cu'].group(1))), 'gamma': gamms, 'kernel': [kernel]}]
    else:
        print "Error specifying C (range) or gamma. Syntax: 'C:5', 'C:5.1', 'C:5_6', 'C:5.1_6.2'."
        exit()
    print param_grid    
else: # For Random search over many grid parameters
    param_grid = [   
    {'C': [0.5, 1, 10, 100, 1000, 1500, 2000], 'kernel': ['poly', 'linear', 'sigmoid'], 'degree': sp_randint(1, 32), 'coef0':sp_randint(1, 5), 'gamma': gammas[op]},
    {'C': [0.5, 1, 10, 100, 1000, 1500, 2000], 'gamma': gammas[op], 'kernel': ['rbf']} ]

if args.N == "auto":
    for p in param_grid:
        p['nu'] = [0.35] #expon(scale=10, loc=0.35) #exp
elif args.N != None:
    for p in param_grid:
        p['nu'] = [float(args.N)]

sys.stderr.write("\n:>> Training settings are OK\n")
sys.stderr.write("Output file: svr_%s_%s_H%s_%s_m%s.out" % (corpus, representation, dimensions, op, min_count))
# Sorted training set:
D = map(list, zip(*sorted(zip(X, y), key=lambda tup:tup[1])))
X = np.array([list(a) for a in D[0]])
if args.t:
    y = map(tener, D[1])
else:
    y = D[1]

del D
for n in xrange(N):
    for params in param_grid:
        if args.N == None:
            svr = SVR()
        else:
            svr = NuSVR()
        if args.k:
            k = int(args.k)
        else:
            k = args.k
        rs = RS(svr, param_distributions = params, n_iter = 10, n_jobs = 24, cv = k)
        try:
            rs.fit(X, y)
        except:
            sys.stderr.write("\n:>> Fitting Error:\n" )

        sys.stderr.write("\n:>> Model selected: %s\n" % (rs.best_params_))        
        f_x = rs.predict(X).tolist()
        sys.stderr.write("\n:>> R2: %s\n" % (r2_score(y, f_x)))
        try:
            num_lines = sum(1 for line in open("svr_%s_%s_H%s_%s_m%s.out" % (corpus, representation, dimensions, op, min_count), "r"))        
        except IOError:
            num_lines = 0

        y_out = {}
        if args.t:
            y_out['estimated_output'] = map(detener, f_x)
        else:
            y_out['estimated_output'] =  f_x

        y_out['best_params'] = rs.best_params_
        y_out['learned_model'] = {'file': "/almac/ignacio/data/svr_models/%s_%s_%s_%s_H%s_%s_m%s.model" % (svr_, corpus, num_lines, representation, dimensions, op, min_count) }
        if args.t:
            y_out['performance'] = r2_score(map(detener, y), map(detener, f_x)) 
        else:
            y_out['performance'] = r2_score(y, f_x)

        with open("svr_%s_%s_H%s_%s_m%s.out" % (corpus, representation, dimensions, op, min_count), "a") as f:
            f.write(str(y_out)+'\n')
        
        joblib.dump(rs, "/almac/ignacio/data/svr_models/%s_%s_%s_%s_H%s_%s_m%s.model" % (svr_, corpus, num_lines, representation, dimensions, op, min_count)) 
with open("sorted_gs_%s.txt" % (corpus), "w") as f:
    for i in map(detener, y):
        f.write(str(i)+'\n')

#sys.stderr.write("\n:>> kernel: %s \n:>> C: %f.4 \n:>> Gamma: %f.4 \n:>> C: %f.4" % (args.K, rs.best_estimator. ))

sys.stderr.write("\n:>> Finished!!\n" )
