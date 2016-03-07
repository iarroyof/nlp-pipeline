import numpy as np
from sklearn.svm import SVR
from sklearn.grid_search import RandomizedSearchCV as RS
from scipy.stats import expon, uniform

inputfile = "/home/ignacio/data/vectors/pairs_headlines_d2v_H300_conc_m5.mtx"
gsfile = "/home/ignacio/data/STS.gs.headlines.txt"
#outputfile = "/home/iarroyof/sem_outputs/svr_output_headlines_100_d2v_conc_300_m5.txt"

X = np.loadtxt(inputfile)
y = np.loadtxt(gsfile)
g = 8.3804

gammas = {'conc': expon(scale=10, loc=8.38049430369), 'sub': expon(scale = 10, loc=15.1454004504), 'convs':expon(scale = 50, loc = 541.113519625)}
op = 'conc'

param_grid = [
   
    {'C': [1, 10, 100, 1000, 1500, 2000], 'kernel': ['poly'], 'degree':[1, 2, 3, 4, 5]},
    {'C': [1, 10, 100, 1000, 1500, 2000], 'gamma': gammas[op], 'kernel': ['rbf']}
 ]

#svr_rbf = SVR(kernel = 'rbf', C = 1e3, gamma = g).fit(X, y)
svr = SVR()
rs = RS(svr, param_distributions=param_grid[0], n_iter=10)
rs.fit(X, y)
#y_rbf = svr_rbf.predict(X)
y_est = rs.predict(X).tolist()

#input = sys.argv[3].split("/")[-1].split(".")[0]
y_out = {}
#y = []
#for o in y_est:
#    y.append(o[0])
y_out['estimated_output'] = y_est
y_out['best_params'] = rs.best_params_
y_out['best_score'] = rs.best_score_
    #y_out['learned_model'] = rs.best_estimator_ 
#with open("nn_output_%s_%s%s_%s_%s_%s_%s.txt"%(input, hidden0, hidden1, units1, units2, lrate, niter), "a") as f:
with open("svr_output_headlines_100_d2v_%s_300_m5.txt" % (op), "a") as f:
    #savetxt(f, y_out)
    f.write(str(y_out)+'\n')

#y={'estimated_output':y_rbf.tolist()}

#with open(outputfile, "w") as f:
#    f.write("%s" % y)
