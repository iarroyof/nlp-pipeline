import numpy as np
from sklearn.svm import SVR, NuSVR
from sklearn.grid_search import RandomizedSearchCV as RS
from scipy.stats import randint as sp_randint
from scipy.stats import expon
from sklearn.externals import joblib
from re import search, M, I
from load_regression import load_regression_data as lr
import sys
#inputfile = "/home/ignacio/data/vectors/pairs_headlines_d2v_H300_sub_m5.mtx"
#inputfile = "/home/iarroyof/data/pairs_headlines_d2v_H300_sub_m5.mtx"

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
parser.add_argument("-m", help="Minimun word frequency taken into account in the corpus. Type '-m m10' for a min_count of 10.", metavar="min_count", default = None)
parser.add_argument("-p", help="Toggle if you will process pairs.", action="store_true", default = False)
parser.add_argument("-N", help="""Toggle if NuSVR (valid only for NuSVR) will be used. The portion of suppoort vectors with respect to the training set 
                                  can be specified in [0,1]. If it is not specified, random values will be generated (normally 
                                  distributed with mean 0.35).""", metavar = "Nu", default = None)
parser.add_argument("-s", help="Toggle if you will process sparse input format.", action="store_true", default = False)
parser.add_argument("-k", help="k-fold cross validation for the randomized search.", metavar="k-fold_cv", default=None)
args = parser.parse_args()
#inputfile = "/home/iarroyof/data/pairs_headlines_d2v_H300_convss_m5.mtx"
#gsfile = "/home/iarroyof/data/STS.gs.headlines.txt"
#outputfile = "/home/iarroyof/sem_outputs/svr_output_headlines_100_d2v_conc_300_m5.txt"
N = int(args.n)

#from pdb import set_trace as st
#st()
try:
    if args.p:
        source = search(r"pairs_(\w+(?:[-|_]\w+)*[0-9]{2,4})_([d2v|w2v|coocc\w*|doc\w*]+)_(H[0-9]{1,4})_([sub|co[nvs{0,2}|rr|nc]+]?)_m([0-9]{1,3}[_[0-9]{0,3}]?)", args.x, M|I)
    else:
        source = search(r"vectors_(\w+(?:[-|_]\w+)*[0-9]{0,4})_(T[0-9]{2,3}[_|-]C[1-9][_|-][0-9]{2})_([d2v|w2v|coocc\w*|doc\w*]+)_([H[0-9]{1,4}]?)_m([0-9]{1,3}[_[0-9]{0,3}]?)", args.x, M|I)            
        #source = search(r"vectors_(\w+(?:[-|_]\w+)*[0-9]{0,4})_([d2v|w2v|coocc\w*|doc\w*]+)_(H[0-9]{1,4})_m([0-9]{1,3}[_[0-9]{0,3}]?)", args.x, M|I)
            # s.group(1) 'headlines13'  s.group(2) 'd2v' s.group(3) 'H300' s.group(4) 'conc'? s.group(5) '5'
    if args.c:
        corpus = args.c
    else:
        #if args.p:
        corpus = source.group(1)
        #else:
            #corpus = source.group(2)
    if args.r:
        representation = args.r
    else:
        if args.p:
            representation = source.group(2)
        else:
            representation = source.group(3)
    if args.d:
        dimensions = args.d
    else:
        if args.d:
            dimensions = source.group(3)[1:]
        else:
            dimensions = source.group(4)[1:]
            #dimensions = source.group(3)[1:]
    if args.m:
        min_count = args.m
    else:
        min_count = source.group(5)
        #min_count = source.group(4)
except IndexError:
    print "\nError in the filename. One or more indicators are missing. Notation: <vectors|pairs>_<source_corpus>_<model_representation>_<Hdimendions>_<''|operation>_<mminimum_count>.mtx\n"
    for i in range(6):
        try:
            print source.group(i)
        except IndexError:
            print ":>> Unparsed: %s" % (i)
            pass
    exit()
except AttributeError:
    print "\nFatal Error in the filename. Notation: <vectors|pairs>_<source_corpus>_<model_representation>_<Hdimendions>_<''|operation>_<mminimum_count>.mtx\n"
    for i in range(6):
        try:
            print source.group(i)
        except AttributeError:
            print ":>> Unparsed: %s" % (i)
            pass            
    exit()

print "\nCorpus: %s\nRepr: %s\nDimms: %s\nF_min: %s\nOpperation: %s\n" % (corpus, representation, dimensions, min_count, source.group(4))
    # TODO: modify the regep for single sentences X file: vectors_d2v_RPM_T17_C1_08_H300_m10.mtx
    

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

# TODO: Is better to find this source mark in the contents of the file, because the name of the file is not secure.
# TODO: Find marks for other input vectors derived from other corpus. It is needed to normalize the names or include source names inside the file.
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
        if infile and infile != "*": # svr_output_headlines100_d2v_convs_300_m5.txt      
            filename = "svr_%s_%s_H%s_predictions.out" % (corpus, representation, dimensions)
            model = joblib.load(op, 'r')
            y_out = {}
            y_out['estimated_output'] = model.predict(X).tolist()
            if args.p:
                y_out['source'] = source.group()+".txt"
            else:
                y_out['source'] = source.group(2)+".txt"
            y_out['model'] = splitext(infile)[0]
        # Add more metadata to the dictionary as required.
            with open(filename, 'a') as f:
                f.write(str(y_out)+'\n')
            sys.stderr.write("\n:>> Output predictions: %s\n" % (filename))
            exit()
        else:
            print "Please specify a file name for loading the SVR pretrained model."            
            exit()
else:
    # example filename: 'pairs_headlines13_d2v_H300_conc_m5.mtx'
    try:
        op = source.group(4)
    except:
        print "Please specify an operation type or give it in the correct way at the filename."
        exit()
        
sys.stderr.write("\n:>> Source: %s\n" % (source.group(1)))


param_grid = [   
    {'C': [1, 10, 100, 1000, 1500, 2000], 'kernel': ['poly', 'linear'], 'degree': sp_randint(1, 32)},
    {'C': [1, 10, 100, 1000, 1500, 2000], 'gamma': gammas[op], 'kernel': ['rbf']} ]

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
        rs = RS(svr, param_distributions = params, n_iter = 10, n_jobs = 8, cv = k)
        try:
            rs.fit(X, y)
        except:
            sys.stderr.write("\n:>> Fitting Error:\n" )

        sys.stderr.write("\n:>> Model selected: %s\n" % (rs.best_params_))        
        f_x = rs.predict(X).tolist()
        sys.stderr.write("\n:>> R2: %s\n" % (rs.best_score_))
        try:
            num_lines = sum(1 for line in open("svr_%s_%s_H%s_%s_m%s.out" % (corpus, representation, dimensions, op, min_count), "r"))        
        except IOError:
            num_lines = 0

        y_out = {}
        y_out['estimated_output'] = f_x
        y_out['best_params'] = rs.best_params_
        y_out['learned_model'] = {'file': "pkl/%s_%s_%s_%s_H%s_%s_m%s.model" % (svr_, corpus, num_lines, representation, dimensions, op, min_count) }
        y_out['performance'] = rs.best_score_

        with open("svr_%s_%s_H%s_%s_m%s.out" % (corpus, representation, dimensions, op, min_count), "a") as f:
            f.write(str(y_out)+'\n')
        
        joblib.dump(rs, "pkl/%s_%s_%s_%s_H%s_%s_m%s.model" % (svr_, corpus, num_lines, representation, dimensions, op, min_count)) 
with open("sorted_gs_%s.txt" % (corpus), "w") as f:
    for i in y:
        f.write(str(i)+'\n')

sys.stderr.write("\n:>> Finished!!\n" )
