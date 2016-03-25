import numpy as np
from sklearn.svm import SVR
from sklearn.grid_search import RandomizedSearchCV as RS
from scipy.stats import randint as sp_randint
from scipy.stats import expon
from sklearn.externals import joblib
from re import search, M, I
#inputfile = "/home/ignacio/data/vectors/pairs_headlines_d2v_H300_sub_m5.mtx"
#inputfile = "/home/iarroyof/data/pairs_headlines_d2v_H300_sub_m5.mtx"

from argparse import ArgumentParser as ap
parser = ap(description='This script trains/applies a SVR over any input dataset of numerical representations. The main aim is to determine a set of learning parameters')
parser.add_argument("-x", help="Input file name (vectors)", metavar="input_file", required=True)
parser.add_argument("-y", help="""Regression labels file. Do not specify this argument if you want to uniauely predict over any test set. In this case, you must to specify
                                the SVR model to be loaded as the parameter of the option -o.""", metavar="regrLabs_file")
parser.add_argument("-n", help="Number of tests to be performed.", metavar="tests_amount", default=1)
parser.add_argument("-o", help="""The operation the input data was derived from. Options: {'conc', 'convs', 'sub'}. In the case you want to give a precalculated center for
                                width randomization, specify the number. e.g. '-o 123.654'. A filename can be specified, which is the file where a SVR model is sotred,
                                e.g. '-o filename.model'""", metavar="operat{or,ion}")
parser.add_argument("-c", help="Sentence corpus over wich the learning will be performed, e.g. '-c headlines13' which indicates headlines (year 2013) corpus.", 
                    metavar="corpus_name", default = None)
parser.add_argument("-r", help="The representation type. options:={w2v, d2v, coocc, doc, docLog, docLogSvd, coocPMI, cooccLogSVD, cooccPMIsvd}.", 
                    metavar="repr_name", default = None)
parser.add_argument("-d", help="Dimensions of the input vector set.", metavar="num_dimensions", default = None)
parser.add_argument("-m", help="Minimun word frequency taken into account in the corpus. Type '-m m10' for a min_count of 10.", metavar="min_count", default = None)
parser.add_argument("-p", help="Toggle if you will process pairs.", action="store_true", default = False)
args = parser.parse_args()
#inputfile = "/home/iarroyof/data/pairs_headlines_d2v_H300_convss_m5.mtx"
#gsfile = "/home/iarroyof/data/STS.gs.headlines.txt"
#outputfile = "/home/iarroyof/sem_outputs/svr_output_headlines_100_d2v_conc_300_m5.txt"
N = int(args.n)
X = np.loadtxt(args.x)

# TODO: Is better to find this source mark in the contents of the file, because the name of the file is not secure.
# TODO: Find marks for other input vectors derived from other corpus. It is needed to normalize the names or include source names inside the file.
gammas = {
        'conc': expon(scale=10, loc=8.38049430369),
        'sub': expon(scale = 20, loc=15.1454004504),
        'convs':expon(scale = 50, loc = 541.113519625) }

if args.y:
    y = np.loadtxt(args.y)

op = args.o

if op.replace('.','',1).isdigit():
    op = 'esp'
    gammas[op] = expon(scale = 20, loc = float(args.o))
    if args.p:
        try:
            source = search(r"[vectors|pairs]+_(\w+(?:[-|_]\w+)*[0-9]{2,4})_([d2v|w2v|coocc\w*|doc\w*]+)_([H[0-9]{1,4}]?)_([sub|co[nvs{0,2}|rr|nc]+]?)_m([0-9]{1,3})", x, M|I)
            # s.group(1) 'headlines13'  s.group(2) 'd2v' s.group(3) 'H300' s.group(4) 'conc' s.group(5) '5'
            if args.c:
                corpus = args.c
            else:
                corpus = source.group(1)
            if args.r:
                representation = args.r
            else:
                representation = source.group(2)
            if args.d:
                dimensions = args.d
            else:
                dimensions = source.group(3)[1:]
            if args.m:
                min_count = args.m
            else:
                min_count = source.group(5)
        except IndexError:
            print "\nError in the filename. One or more indicators are missing. Notation: <vectors|pairs>_<source_corpus>_<model_representation>_<Hdimendions>_<''|operation>_<mminimum_count>.mtx\n"
            exit()
        except AttributeError:
            print "\nFatal Error in the filename. Notation: <vectors|pairs>_<source_corpus>_<model_representation>_<Hdimendions>_<''|operation>_<mminimum_count>.mtx\n"
            exit()
        print "\nCorpus: %s \nRepr: %s \nDimms: %s \n F_min: %s\n" % (corpus, representation, dimensions, min_count)
    else:
        source = search(r"T[0-9]{2}_C[1-9]_[0-9]{2}", args.x, M|I)
elif not op in gammas:
    from os.path import basename, splitext
    import sys 
    # example filename: 'pairs_headlines13_d2v_H300_conc_m5.mtx'
    if args.p:
        try:
            source = search(r"[vectors|pairs]+_(\w+(?:[-|_]\w+)*[0-9]{2,4})_([d2v|w2v|coocc\w*|doc\w*]+)_([H[0-9]{1,4}]?)_([sub|co[nvs{0,2}|rr|nc]+]?)_m([0-9]{1,3})", args.x, M|I)
            if args.c:
                corpus = args.c
            else:
                corpus = source.group(1)
            if args.r:
                representation = args.r
            else:
                representation = source.group(2)
            if args.d:
                dimensions = args.d
            else:
                dimensions = source.group(3)[1:]
            if args.m:
                min_count = args.m
            else:
                min_count = source.group(5)
        except IndexError:
            print "\nError in the filename. One or more indicators are missing. Notation: <vectors|pairs>_<source_corpus>_<model_representation>_<Hdimendions>_<''|operation>_<mminimum_count>.mtx\n"
            exit()
        except AttributeError:
            print "\nFatal Error in the filename. Notation: <vectors|pairs>_<source_corpus>_<model_representation>_<Hdimendions>_<''|operation>_<mminimum_count>.mtx\n"
            exit()
    else:
        source = search(r"T[0-9]{2}_C[1-9]_[0-9]{2}", args.x, M|I)
        
    sys.stderr.write("\n:>> Source: %s" % (source.group(1)))
    infile = basename(op) # SVR model file name
    if infile and infile != "*":       
        filename = splitext(infile)[0]+'_predictions.out'
        model = joblib.load(op, 'r')
        #print ":>> Model loaded from:", op
        y_out = {}
        y_out['estimated_output'] = model.predict(X).tolist()
        y_out['source'] = source.group()+".txt"
        # Add more metadata to the dictionary as required.
        with open(filename, 'a') as f:
            f.write(str(y_out)+'\n')
        exit()
    else:
        print "Please specify a file name for loading the SVR pretrained model."            
        exit()
else:
    import sys 
    # example filename: 'pairs_headlines13_d2v_H300_conc_m5.mtx'
    from pdb import set_trace as st
#    st()
    if args.p:
        try:
            source = search(r"[vectors|pairs]+_(\w+(?:[-|_]\w+)*[0-9]{2,4})_([d2v|w2v|coocc\w*|doc\w*]+)_([H[0-9]{1,4}]?)_([sub|co[nvs{0,2}|rr|nc]+]?)_m([0-9]{1,3})", args.x, M|I)
            if args.c:
                corpus = args.c
            else:
                corpus = source.group(1)
            if args.r:
                representation = args.r
            else:
                representation = source.group(2)
            if args.d:
                dimensions = args.d
            else:
                dimensions = source.group(3)[1:]
            if args.m:
                min_count = args.m
            else:
                min_count = source.group(5)
        except IndexError:
            print "\nError in the filename. One or more indicators are missing. Notation: <vectors|pairs>_<source_corpus>_<model_representation>_<Hdimendions>_<''|operation>_<mminimum_count>.mtx\n"
            exit()
        except AttributeError:
            print "\nFatal Error in the filename. Notation: <vectors|pairs>_<source_corpus>_<model_representation>_<Hdimendions>_<''|operation>_<mminimum_count>.mtx\n"
            exit()
    else:
        source = search(r"T[0-9]{2}_C[1-9]_[0-9]{2}", args.x, M|I)
        
    sys.stderr.write("\n:>> Source: %s" % (source.group(1)))


param_grid = [   
    {'C': [1, 10, 100, 1000, 1500, 2000], 'kernel': ['poly'], 'degree': sp_randint(1, 5)},
    {'C': [1, 10, 100, 1000, 1500, 2000], 'gamma': gammas[op], 'kernel': ['rbf']} ]

for n in xrange(N):
    for params in param_grid:
        svr = SVR()
        rs = RS(svr, param_distributions = params, n_iter = 10, n_jobs = 24)
        rs.fit(X, y)
        f_x = rs.predict(X).tolist()
        try:
            num_lines = sum(1 for line in open("svr_%s_%s_H%s_%s_m%s.out" % (corpus, representation, dimensions, op, min_count), "r"))        
        except IOError:
            num_lines = 0

        y_out = {}
        y_out['estimated_output'] = f_x
        y_out['best_params'] = rs.best_params_
        y_out['learned_model'] = {'file': "pkl/svr_%s_%s_%s_H%s_%s_m%s.model" % (corpus, num_lines, representation, dimensions, op, min_count) }
        y_out['performance'] = rs.best_score_

        with open("svr_%s_%s_H%s_%s_m%s.out" % (corpus, representation, dimensions, op, min_count), "a") as f:
            f.write(str(y_out)+'\n')
        
        joblib.dump(rs, "pkl/svr_%s_%s_%s_H%s_%s_m%s.model" % (corpus, num_lines, representation, dimensions, op, min_count)) 
