from sknn.platform import cpu64, threading
from scipy import stats
from sklearn.grid_search import RandomizedSearchCV

from sknn.mlp import Regressor, Layer
from numpy import  sqrt, loadtxt, savetxt, array
import sys

if len(sys.argv) < 4:
    print "Usage: python nn.py train_vectors scores test_vectors searches"
    exit()

hidden0 = "Tanh"#"Sigmoid"
hidden1 = "Tanh"
hidden2 = "Tanh"
output = "Sigmoid"
lrate = 0.01
niter = 100
units1 = 30
units2 = 10

# Pyramid rule: for 1-hidden layer: sqrt(m*n); 'm' input nodes and 'n' output neurons. For 2-hidden layers: 
# r = cubicroot(n/m); n_hidden0 = m*r^2 and n_hidden1 = m*r

X_train = loadtxt(sys.argv[1])
#print type(X_train)
y_train = loadtxt(sys.argv[2])
#print type(y_train)
X_valid = loadtxt(sys.argv[3])
#print type(X_valid)
y_valid = loadtxt(sys.argv[4])
try:
    N = int(sys.argv[5]) # The number of searches
except IndexError:
    N = 1
#print "X_train: %s"%(str(X_train.shape))
#print "y_train: %s"%(str(y_train.shape))
#print "X_test: %s"%(str(X_valid.shape))

params = {
    'learning_momentum': stats.uniform(0.01, 1.5),
    'learning_rate': stats.uniform(0.001, 0.05),
    'learning_rule': ['sgd', 'adagrad', 'rmsprop'], #'momentum', 'nesterov', 'adadelta', 'adagrad', 'rmsprop'],
    'regularize': ["L1", "L2", None],
    'hidden0__units': stats.randint(2, 20),
    'hidden0__type': ["Rectifier", "Sigmoid", "Tanh"],
    'hidden1__units': stats.randint(2, 20),
    'hidden1__type': ["Rectifier", "Sigmoid", "Tanh"],
    'hidden2__units': stats.randint(2, 20),
    'hidden2__type': ["Rectifier", "Sigmoid", "Tanh"],
    'output__type': ["Linear"] }

nn = Regressor(
    layers=[
        Layer(hidden0, units=1),
        Layer(hidden1, units=1),
        Layer(hidden2, units=1),
        Layer(output, units=1)],
    learning_rate=lrate,
    n_iter=niter)

parameter_grid = []
for i in xrange(N):
    parameter_grid.append(params)    

if len(X_train) != len(y_train):
    sys.stderr.write("Number of samples and number of labels do not match.")
    exit()

from /home/ignacio/distributionalSemanticStabilityThesis/mkl_regressor import test_predict

for params in parameter_grid:
    crash = True
    try:
        rs = RandomizedSearchCV(nn, n_iter = 10, n_jobs = 4, param_distributions=params, scoring="mean_squared_error")
        rs.fit(X_train, y_train)
        crash = False
    except RuntimeError:
        sys.stderr.write("--------------------- [Crashed by RunTimeERROR] --------------------- \n %s\n" % params)
        pass
    y = test_predict(data = X_valid, machine = rs.best_estimator_, labels = y_valid, graph = True)    
    
    #y_out = {}
    #y_out['estimated_output'] = y
    #y_out['best_params'] = rs.best_params_
    #y_out['performance'] = rs.best_score_

    #with open("nn_output_headlines_30_d2v_conv_300_m5.txt", "a") as f:
    #    f.write(str(y_out)+'\n')
