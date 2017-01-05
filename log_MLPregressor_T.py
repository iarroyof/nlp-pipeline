"""
This tutorial introduces the multilayer perceptron using Theano.

 A multilayer perceptron is a logistic regressor where
instead of feeding the input to the logistic regression you insert a
intermediate layer, called the hidden layer, that has a nonlinear
activation function (usually tanh or sigmoid) . One can use many such
hidden layers making the architecture deep. The tutorial will also tackle
the problem of MNIST digit classification.

.. math::

    f(x) = G( b^{(2)} + W^{(2)}( s( b^{(1)} + W^{(1)} x))),

References:

    - textbooks: "Pattern Recognition and Machine Learning" -
                 Christopher M. Bishop, section 5

"""

from __future__ import print_function

__docformat__ = 'restructedtext en'

__op = "conc"

import os
import sys
import timeit

import numpy

import theano
import theano.tensor as T

from logistic_sgd import LogisticRegression, load_data

from data_theano import *
import dill
# start-snippet-1
class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None,
                 activation=T.tanh):
        """
        Typical hidden layer of a MLP: units are fully-connected and have
        sigmoidal activation function. Weight matrix W is of shape (n_in,n_out)
        and the bias vector b is of shape (n_out,).

        NOTE : The nonlinearity used here is tanh

        Hidden unit activation is given by: tanh(dot(input,W) + b)

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dmatrix
        :param input: a symbolic tensor of shape (n_examples, n_in)

        :type n_in: int
        :param n_in: dimensionality of input

        :type n_out: int
        :param n_out: number of hidden units

        :type activation: theano.Op or function
        :param activation: Non linearity to be applied in the hidden
                           layer
        """
        self.input = input
        # end-snippet-1

        # `W` is initialized with `W_values` which is uniformely sampled
        # from sqrt(-6./(n_in+n_hidden)) and sqrt(6./(n_in+n_hidden))
        # for tanh activation function
        # the output of uniform if converted using asarray to dtype
        # theano.config.floatX so that the code is runable on GPU
        # Note : optimal initialization of weights is dependent on the
        #        activation function used (among other things).
        #        For example, results presented in [Xavier10] suggest that you
        #        should use 4 times larger initial weights for sigmoid
        #        compared to tanh
        #        We have no info for other function, so we use the same as
        #        tanh.
        if W is None:
            #low = -numpy.sqrt(6. / (n_in + n_out))
            #high = numpy.sqrt(6. / (n_in + n_out))
            low = -2.0
            high = 2.0
            Weights = rng.uniform(low=low, high=high, size=(n_in, n_out))
            W_values = numpy.asarray(Weights, dtype=theano.config.floatX)

            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b

        lin_output = T.dot(input, self.W) + self.b
        self.output = (
            lin_output if activation is None
            else activation(lin_output)
        )
        # parameters of the model
        self.params = [self.W, self.b]

# start-snippet-2
class MLP(object):
    """Multi-Layer Perceptron Class

    A multilayer perceptron is a feedforward artificial neural network model
    that has one layer or more of hidden units and nonlinear activations.
    Intermediate layers usually have as activation function tanh or the
    sigmoid function (defined here by a ``HiddenLayer`` class)  while the
    top layer is a softmax layer (defined here by a ``LogisticRegression``
    class).
    """

    def __init__(self, rng, input, n_in, n_hidden, n_out):
        """Initialize the parameters for the multilayer perceptron

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
        architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
        which the datapoints lie

        :type n_hidden: int
        :param n_hidden: number of hidden units

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
        which the labels lie

        """

        # Since we are dealing with a one hidden layer MLP, this will translate
        # into a HiddenLayer with a tanh activation function connected to the
        # LogisticRegression layer; the activation function can be replaced by
        # sigmoid or any other nonlinear function
#        self.hiddenLayer = HiddenLayer(
#            rng=rng,
#            input=input,
#            n_in=n_in,
#            n_out=n_hidden,
#            activation=T.tanh
#        )

        self.hiddenLayers=[] 
        for i in xrange(len(n_hidden)):
            if i == 0:
                self.hiddenLayers.append(HiddenLayer(
                rng=rng,
                input=input, 
                n_in=n_in,       # [dim(input), n_hiden[0],...]
                n_out=n_hidden[i],  # [n_hidden[0], n_hidden[1],...]
                activation=T.tanh)
                )
            elif i < len(n_hidden) and i > 0:
                self.hiddenLayers.append(HiddenLayer(
                rng=rng,
                input=self.hiddenLayers[i-1].output,           
                n_in=n_hidden[i-1],       # [dim(input), n_hiden[0],...]
                n_out=n_hidden[i],  # [n_hidden[0], n_hidden[1],...]
                activation=T.tanh)
                )
        
        # The logistic regression layer gets as input the hidden units
        # of the hidden layer
        self.logRegressionLayer = LogisticRegression(
            input=self.hiddenLayers[-1].output,
            n_in=n_hidden[-1],
            n_out=n_out
        )
        # end-snippet-2 start-snippet-3
        # L1 norm ; one regularization option is to enforce L1 norm to
        # be small

        self.L1 = (
            numpy.array([abs(self.hiddenLayers[i].W).sum() for i in xrange(len(n_hidden))]).sum()
            #abs(self.hiddenLayer.W).sum()
            + abs(self.logRegressionLayer.W).sum()
        )

        # square of L2 norm ; one regularization option is to enforce
        # square of L2 norm to be small
        self.L2_sqr = (
            numpy.array([abs(self.hiddenLayers[i].W ** 2).sum() for i in xrange(len(n_hidden))]).sum()
            #(self.hiddenLayer.W ** 2).sum()
            + (self.logRegressionLayer.W ** 2).sum()
        )

        # negative log likelihood of the MLP is given by the negative
        # log likelihood of the output of the model, computed in the
        # logistic regression layer
        self.negative_log_likelihood = (
            self.logRegressionLayer.negative_log_likelihood
        )
        # same holds for the function computing the number of errors
        self.errors = self.logRegressionLayer.errors
        self.y_pred = self.logRegressionLayer.y_pred
        # the parameters of the model are the parameters of the two layer it is
        # made out of
        self.params = []
        for i in xrange(len(n_hidden)):
            self.params += self.hiddenLayers[i].params

        self.params = self.params + self.logRegressionLayer.params
        # end-snippet-3

        # keep track of model input
        self.input = input


def test_mlp(learning_rate=0.01, L1_reg=0.00, L2_reg=0.0001, n_epochs=1000,
             dataset='mnist.pkl.gz', batch_size=20, n_hidden=[10], verbose = False):
    """
    Demonstrate stochastic gradient descent optimization for a multilayer
    perceptron

    This is demonstrated on MNIST.

    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
    gradient

    :type L1_reg: float
    :param L1_reg: L1-norm's weight when added to the cost (see
    regularization)

    :type L2_reg: float
    :param L2_reg: L2-norm's weight when added to the cost (see
    regularization)

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

    :type dataset: string
    :param dataset: the path of the MNIST dataset file from
                 http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz


   """
    dim = dataset
    path = "/almac/ignacio/data/sts_all/"
    tr_px = path + "pairs-SI/vectors_H%s/pairs_eng-SI-test-2e6-nonempty_d2v_H%s_%s_m5w8.mtx" % (dim, dim, __op)
    tr_py = path + "pairs-SI/STS.gs.all-eng-SI-test-nonempty.txt"
    ts_px = path + "pairs-NO/vectors_H%s/pairs_eng-NO-test-2e6-nonempty_d2v_H%s_%s_m5w8.mtx.half0" % (dim, dim, __op)
    ts_py = path + "pairs-NO/STS.gs.all-eng-NO-test-nonempty-half0.txt"
    vl_px = path + "pairs-NO/vectors_H%s/pairs_eng-NO-test-2e6-nonempty_d2v_H%s_%s_m5w8.mtx.half1" % (dim, dim, __op)
    vl_py = path + "pairs-NO/STS.gs.all-eng-NO-test-nonempty-half1.txt"
    #print ("%s\n%s\n%s\n%s\n%s\n%s\n" % (tr_px,tr_py,ts_px,ts_py,vl_px,vl_py))
    #tr_px = path + "pairs-NO_2013/vectors_H"+ dataset+"/pairs_eng-NO-test-2e6-nonempty_OnWN_d2v_H"+ dataset+"_sub_m5w8.mtx"
    #tr_py = path + "pairs-NO_2013/STS.gs.OnWN.txt"
    #ts_px = path + "pairs-SI_2014/vectors_H"+ dataset+"/pairs_eng-NO-test-2e6-nonempty_deft-news_d2v_H"+ dataset+"_sub_m5w8.mtx"
    #ts_py = path + "pairs-SI_2014/STS.gs.deft-news.txt"
    #vl_px = path + "pairs-NO_2013/vectors_H"+ dataset+"/pairs_eng-NO-test-2e6-nonempty_FNWN_d2v_H"+ dataset+"_sub_m5w8.mtx"
    #vl_py = path + "pairs-NO_2013/STS.gs.FNWN.txt"

    datasets = load_my_data(tr_px,tr_py,ts_px,ts_py,vl_px,vl_py, shared=True)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]
    
    train_samples = train_set_x.get_value(borrow=True).shape[0]
    n_train_batches = train_samples // batch_size
    valid_samples = valid_set_x.get_value(borrow=True).shape[0]
    n_valid_batches = valid_samples // batch_size
    test_samples = test_set_x.get_value(borrow=True).shape[0]
    n_test_batches = test_samples // batch_size
    in_dimensions = train_set_x.get_value(borrow=True).shape[1]
    
    #print("Shape train %s, test %s, validation %s" % (train_set_x.get_value(borrow=True).shape, test_set_x.get_value(borrow=True).shape, valid_set_x.get_value(borrow=True).shape))
    
    assert train_set_x.get_value(borrow=True).shape[1] == valid_set_x.get_value(borrow=True).shape[1] == test_set_x.get_value(borrow=True).shape[1] # verify dimensions

    #k_classes = max(train_set_y) + 1 # for 0-based class index
    k_classes = T.max(train_set_y, axis=0).eval() + 1

    ######################
    # BUILD ACTUAL MODEL #
    ######################

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch
    x = T.matrix('x')  # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels

    rng = numpy.random.RandomState(1234)

    # construct the MLP class
    classifier = MLP(
        rng=rng,
        input=x,
        n_in=in_dimensions,
        n_hidden=n_hidden,
        n_out=k_classes
    )

    # start-snippet-4
    # the cost we minimize during training is the negative log likelihood of
    # the model plus the regularization terms (L1 and L2); cost is expressed
    # here symbolically
    cost = (
        classifier.negative_log_likelihood(y)
        + L1_reg * classifier.L1
        + L2_reg * classifier.L2_sqr
    )
    # end-snippet-4

    # compiling a Theano function that computes the mistakes that are made
    # by the model on a minibatch
    test_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: test_set_x[index * batch_size:(index + 1) * batch_size],
            y: test_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: valid_set_x[index * batch_size:(index + 1) * batch_size],
            y: valid_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )

    # start-snippet-5
    # compute the gradient of cost with respect to theta (sorted in params)
    # the resulting gradients will be stored in a list gparams
    gparams = [T.grad(cost, param) for param in classifier.params]

    # specify how to update the parameters of the model as a list of
    # (variable, update expression) pairs

    # given two lists of the same length, A = [a1, a2, a3, a4] and
    # B = [b1, b2, b3, b4], zip generates a list C of same size, where each
    # element is a pair formed from the two lists :
    #    C = [(a1, b1), (a2, b2), (a3, b3), (a4, b4)]
    updates = [
        (param, param - learning_rate * gparam)
        for param, gparam in zip(classifier.params, gparams)
    ]

    # compiling a Theano function `train_model` that returns the cost, but
    # in the same time updates the parameter of the model based on the rules
    # defined in `updates`
    train_model = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )
    # end-snippet-5

    ###############
    # TRAIN MODEL #
    ###############

    # early-stopping parameters
    patience = 10000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
                           # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                     # considered significant
    validation_frequency = min(n_train_batches, patience // 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0.
    start_time = timeit.default_timer()

    epoch = 0
    done_looping = False
    
    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in range(n_train_batches):
          
            minibatch_avg_cost = train_model(minibatch_index)
            # iteration number
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:
                # compute zero-one loss on validation set
                validation_losses = [validate_model(i) for i in range(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)
                if verbose:
                    print(
                        'epoch %i, minibatch %i/%i, validation error %f %%' %
                        (
                        epoch,
                        minibatch_index + 1,
                        n_train_batches,
                        this_validation_loss * 100.
                        )
                    )

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:
                    #improve patience if loss improvement is good enough
                    if (
                        this_validation_loss < best_validation_loss *
                        improvement_threshold
                    ):
                        patience = max(patience, iter * patience_increase)

                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    # test it on the test set
                    test_losses = [test_model(i) for i
                                   in range(n_test_batches)]
                    test_score = numpy.mean(test_losses)
                    if verbose:
                        print(('     epoch %i, minibatch %i/%i, test error of '
                          'best model %f %%') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           test_score * 100.))

            if patience <= iter:
                done_looping = True
                break

    end_time = timeit.default_timer()
    print('Run time required: %.2fm' % ((end_time - start_time) / 60.))

    return best_iter + 1, best_validation_loss * 100.0, test_score * 100.0, classifier

def predict(model, dim):
    """
    An example of how to load a trained model and use it
    to predict labels.
    """
    # We can test it on some examples from test test
    path = "/almac/ignacio/data/sts_all/"
    tr_px = path + "pairs-SI/vectors_H%s/pairs_eng-SI-test-2e6-nonempty_d2v_H%s_%s_m5w8.mtx" % (dim, dim, __op)
    tr_py = path + "pairs-SI/STS.gs.all-eng-SI-test-nonempty.txt"
    ts_px = path + "pairs-NO/vectors_H%s/pairs_eng-NO-test-2e6-nonempty_d2v_H%s_%s_m5w8.mtx.half0" % (dim, dim, __op)
    ts_py = path + "pairs-NO/STS.gs.all-eng-NO-test-nonempty-half0.txt"
    vl_px = path + "pairs-NO/vectors_H%s/pairs_eng-NO-test-2e6-nonempty_d2v_H%s_%s_m5w8.mtx.half1" % (dim, dim, __op)
    vl_py = path + "pairs-NO/STS.gs.all-eng-NO-test-nonempty-half1.txt"
    # /almac/ignacio/data/sts_all/pairs-NO/STS.gs.all-eng-NO-test-nonempty-half0.txt
    # load the saved model
    with open(model, 'rb') as f:
        classifier = dill.load(f)

    # compile a predictor function
    predict_model = theano.function(
        inputs=[classifier.input],
        outputs=classifier.y_pred
        )

    datasets = load_my_data(tr_px,tr_py,ts_px,ts_py,vl_px,vl_py, shared=True)

    test_set_x, test_set_y = datasets[2]

    test_set_x = test_set_x.get_value()

    predicted_values = predict_model(test_set_x)
    return predicted_values

if __name__ == '__main__':
    from ast import literal_eval
    from argparse import ArgumentParser as ap
    parser = ap(description='This script trains/applies a Multi-Layer Perceptron over any input dataset of numerical representations. The main aim is to determine a set of learning parameters and architecture.')
    parser.add_argument("--hidden", help="Size of the hidden layer", metavar="hidden", default=100)
    parser.add_argument("--dims", help="Size of the input layer", metavar="dims", default=2)
    parser.add_argument("--lrate", help="The learning rate", metavar="lrate", default=0.01)
    parser.add_argument("--predict", help="Predict by loading an existent mode or train a new model (specify the file name of the trained model)", metavar="predict", default=None)
    parser.add_argument("--l1_reg", help="L1 regularization parameter", metavar="l1_reg", default=0.0)
    parser.add_argument("--l2_reg", help="L2 regularization parameter", metavar="l2_reg", default=0.00010)
    parser.add_argument("--n_epochs", help="Maximum number of training epochs", metavar="n_epochs", default=1000)
    parser.add_argument("--batch", help="Size of the training mini batch", metavar="batch", default=20)
    parser.add_argument("--save", help="Toggles whether you want to save the learned model", action="store_true")
    args = parser.parse_args()
    
    
    if not args.predict:
        # learning_rate=0.01, L1_reg=0.00, L2_reg=0.0001, n_epochs=1000, dataset='mnist.pkl.gz', batch_size=20, n_hidden=500)
        best_iter, best_validation_loss, test_score, model = test_mlp(learning_rate=float(args.lrate), batch_size=20, 
                                                                      n_epochs=int(args.n_epochs), n_hidden=literal_eval(args.hidden), 
                                                                      dataset=int(args.dims), verbose=True, 
                                                                      L1_reg=float(args.l1_reg), L2_reg=float(args.l2_reg) )
        if args.save:
            with open("mlp_STS-all_H%s_idim%s.pkl" % (args.hidden, args.dims), 'wb') as f:
                dill.dump(model, f)

        with open("mlp.out", "a") as f:
            f.write("Best validation score of %f %% obtained at iteration %i, with test performance %f %%\tParameters: dims = %d\tHidden = %s\n" % (best_validation_loss, best_iter, test_score, int(args.dims), args.hidden))

        print("Best validation score of %f %% obtained at iteration %i, with test performance %f %%\tParameters: dims = %d\tHidden = %s\n" % (best_validation_loss, best_iter, test_score, int(args.dims), args.hidden))

    else:
        #with open("mlp.predict", "w") as f:
        y_pred = predict(args.predict, int(args.dims))
        for item in y_pred:
            #thefile.write("%s\n" % item)    
            print (item)
