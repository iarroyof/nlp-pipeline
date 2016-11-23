from sklearn.decomposition import TruncatedSVD
from sklearn import preprocessing
from argparse import ArgumentParser as ap
from numpy import loadtxt, savetxt
from numpy import array
parser = ap()
parser.add_argument("-x", help="Input file name (vectors)", metavar="input_file", required=True)
parser.add_argument("-n", help="N desired output components", metavar="n_components", required=True)
parser.add_argument("-s", help="Toggle save computed matrices.", default=False, action='store_true')
args = parser.parse_args()

fout = args.x+".svd"
foutN = args.x+".norm"
X = loadtxt(args.x)
#print "Mean before rescaling: %s" % (X.mean(axis=0))
#print "Var before rescaling : %s" % (X.std(axis=0))
X = preprocessing.scale(X)

print "Origin shape:", X.shape

if args.s:
    savetxt(foutN, X)
#print "Mean after: %s" % (X.mean(axis=0))
#print "Var afeter: %s" % (X.std(axis=0))
svd = TruncatedSVD(n_components=int(args.n), random_state=42)
X_p = svd.fit_transform(X)
print "Final shape:",X_p.shape

print "Var_ratio: %f percent" % (svd.explained_variance_ratio_.sum()*100)
print "Variance : %s" % (svd.explained_variance_)

if args.s:
    savetxt(fout, X_p)
