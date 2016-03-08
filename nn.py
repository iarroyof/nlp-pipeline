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
lrate = 0.001
niter = 100
units1 = 30
units2 = 10

X_train = loadtxt(sys.argv[1])
#print type(X_train)
y_train = loadtxt(sys.argv[2])
#print type(y_train)
X_valid = loadtxt(sys.argv[3])
#print type(X_valid)
try:
    N = int(sys.argv[4]) # The number of searches
except IndexError:
    N = 1
#print "X_train: %s"%(str(X_train.shape))
#print "y_train: %s"%(str(y_train.shape))
#print "X_test: %s"%(str(X_valid.shape))

nn = Regressor(
    layers=[
        Layer(hidden2, units=1),
        Layer(hidden1, units=1),
        Layer(hidden0, units=1),
        Layer(output)],
    learning_rate=lrate,
    n_iter=niter)

rs = RandomizedSearchCV(nn, n_iter = 10, n_jobs = 4, param_distributions={
    'learning_momentum': stats.uniform(0.1, 1.5),
    'learning_rate': stats.uniform(0.009, 0.1),
    'learning_rule': ['sgd'], #'momentum', 'nesterov', 'adadelta', 'adagrad', 'rmsprop'],
    'regularize': ["L1", "L2", None],
    'hidden0__units': stats.randint(2, 300),
    'hidden0__type': ["Rectifier", "Sigmoid", "Tanh"],
    'hidden1__units': stats.randint(2, 300),
    'hidden1__type': ["Rectifier", "Sigmoid", "Tanh"],
    'hidden2__units': stats.randint(4, 300),
    'hidden2__type': ["Rectifier", "Sigmoid", "Tanh"],
    'output__type': ["Linear", "Softmax"]})
    
#rs.fit(a_in, a_out)    
if len(X_train) != len(y_train):
    sys.stderr.write("Number of samples and number of labels do not match.")
    exit()

for t in xrange(N):
    crash = True
    while(crash):
        try:
            rs.fit(X_train, y_train)
            crash = False
        except RuntimeError:
            sys.stderr.write("--------------------- [Crashed by RunTimeERROR. restarting] --------------------- \n")
            crash = True
    
    sys.stderr.write("Best Parameters: %s, score: %s\n" % (str(rs.best_params_), str(rs.best_score_)))
    y_ = rs.predict(X_valid)
    y = []
    for o in y_:
        y.append(o[0])
    
    input = sys.argv[3].split("/")[-1].split(".")[0]
    y_out = {}
    y_out['estimated_output'] = y
    y_out['best_params'] = rs.best_params_
    y_out['best_score'] = rs.best_score_

    with open("nn_output_headlines_30_d2v_conv_300_m5.txt", "a") as f:
        f.write(str(y_out)+'\n')
    
