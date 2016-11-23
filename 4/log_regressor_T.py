# Example from Theano tutorial: http://deeplearning.net/software/theano/tutorial/examples.html#a-real-example-logistic-regression
import numpy
import theano
import theano.tensor as T
from data_theano import *

rng = numpy.random

N = 10                                   # training sample size
feats = 10                               # number of input variables

path = "/almac/ignacio/data/sts_all/"
tr_px = path + "pairs-NO_2013/vectors_H10/pairs_eng-NO-test-2e6-nonempty_OnWN_d2v_H10_sub_m5w8.mtx"
tr_py = path + "pairs-NO_2013/STS.gs.OnWN.txt"
ts_px = path + "pairs-SI_2014/vectors_H10/pairs_eng-NO-test-2e6-nonempty_deft-news_d2v_H10_sub_m5w8.mtx"
ts_py = path + "pairs-SI_2014/STS.gs.deft-news.txt"
vl_px = path + "pairs-NO_2013/vectors_H10/pairs_eng-NO-test-2e6-nonempty_FNWN_d2v_H10_sub_m5w8.mtx"
vl_py = path + "pairs-NO_2013/STS.gs.FNWN.txt"

# generate a dataset: D = (input_values, target_class)
#D = (rng.randn(N, feats), rng.randint(size=N, low=0, high=2))
# [(train_set_x, train_set_y), (valid_set_x, valid_set_y), (test_set_x, test_set_y)]
D = load_my_data(tr_px,tr_py,ts_px,ts_py,vl_px,vl_py)[0]
print "Shapes: D[0]: %s,  D[1]: %s" % (D[0].shape, D[1].shape)

training_steps = 10000
feats =  D[0].shape[1]
assert D[0].shape[0] == D[1].shape[0]
# Declare Theano symbolic variables
x = T.dmatrix("x")
y = T.dvector("y")

# initialize the weight vector w randomly
#
# this and the following bias variable b
# are shared so they keep their values
# between training iterations (updates)
w = theano.shared(rng.randn(feats), name="w")

# initialize the bias term
b = theano.shared(0., name="b")

print("Initial model:")
print(w.get_value())
print(b.get_value())

# Construct Theano expression graph
p_1 = 1 / (1 + T.exp(-T.dot(x, w) - b))   # Probability that target = 1
prediction = p_1 > 0.5                    # The prediction thresholded
xent = -y * T.log(p_1) - (1-y) * T.log(1-p_1) # Cross-entropy loss function
cost = xent.mean() + 0.01 * (w ** 2).sum()# The cost to minimize
gw, gb = T.grad(cost, [w, b])             # Compute the gradient of the cost
                                          # w.r.t weight vector w and
                                          # bias term b
                                          # (we shall return to this in a
                                          # following section of this tutorial)

# Compile
train = theano.function(
          inputs=[x,y],
          outputs=[prediction, xent],
          updates=((w, w - 0.1 * gw), (b, b - 0.1 * gb)))
predict = theano.function(inputs=[x], outputs=prediction)

# Train
for i in range(training_steps):
    pred, err = train(D[0], D[1])

print("Final model:")
print(w.get_value())
print(b.get_value())
print("target values for D:")
print(D[1])
print("prediction on D:")
print(predict(D[0]))
