"""
Experimental machine for Reproducing kernel Hilbert spaces implemented in Theano.
Main functionalities of this code were aquired from the Theano Multilayer Perceptron
tutorial.

Only RKHS and number of layer functionalities were introduced me.
"""

from __future__ import print_function
__docformat__ = 'restructedtext en'

__op = "conc"

import os
import sys
import timeit
import numpy
import dill
import theano
import theano.tensor as T
from theano.tensor import _tensor_py_operators as ops
from logistic_sgd import LogisticRegression, load_data
from data_theano import *

class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, batch_s, W=None, s=None, b=None,
                 kernel="gaussian"):
        """
        Typical hidden layer of a mlRKHS: units are fully-connected and have
        not activation function. Weight (mean) matrix W is of shape (n_in,n_out)
        and the bias vector b is of shape (n_out,).

        NOTE : The nonlinearity used here is tanh when kernel product is not asked

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dmatrix
        :param input: a symbolic tensor of shape (n_examples, n_in)

        :type n_in: int
        :param n_in: dimensionality of input

        :type n_out: int
        :param n_out: number of hidden units

        :type kernel: str
        :param kernel: Kernel type asked by user
        """
        self.input = input
        self.kernel = kernel

        if W is None:
            #low = -numpy.sqrt(6. / (n_in + n_out))
            #high = numpy.sqrt(6. / (n_in + n_out))
            low = -5.0
            high = 5.0
            Weights = rng.uniform(low=low, high=high, size=(n_in, n_out))
            W_values = numpy.asarray(Weights, dtype=theano.config.floatX)
            W = theano.shared(value=W_values, name='W', borrow=True)
        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)
        if s is None:
            sigmas = rng.uniform(low=0.001, high=100, size=(n_out,))
            s_values = numpy.asarray(sigmas, dtype=theano.config.floatX)
            s = theano.shared(value=s_values, name='s', borrow=True)

        self.W = W
        self.b = b
        self.s = s
        output = 0

        if self.kernel is None:
            output = T.dot(input, self.W) + self.b
        elif self.kernel == "sigmoid":
            output = T.tanh(T.dot(input, self.W) + self.b)
        elif self.kernel == "gaussian":
            # The RKHS inner product via the Gaussian kernel (dot_H)
            h_values = numpy.zeros((batch_s, n_out), dtype=theano.config.floatX)
            dot_H = theano.shared(value=h_values, name='dot_H', borrow=True)
            for i in range(batch_s):
                T.set_subtensor(dot_H[i:],theano.scan(lambda w, sig, bias: \
                               T.exp(-ops.norm(w - input[i], 2) ** 2 / 2*sig ** 2) + bias,
                               sequences=[self.W.T, self.s, self.b])[0])
            output = dot_H.get_value()

        self.output = output
        # parameters of this hidden layer
        self.params = [self.W, self.b, self.s]

class mlRKHS(object):
    """Multi-Layer Reproducing Kernel Hilbert Spaces Class
    """

    def __init__(self, rng, input, n_in, n_hidden, n_out, batch_s, kernel=None):
        """Initialize the parameters for the multilayer RKHS

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
        architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
        which the datapoints lie

        :type n_hidden: list of int
        :param n_hidden: number of hidden units for each layer

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
        which the labels lie

        """
        # Constructor of the hidden layers according to the number and size required of them.
        self.hiddenLayers=[] 
        for i in xrange(len(n_hidden)):
            if i == 0:
                self.hiddenLayers.append(HiddenLayer(
                rng=rng,
                input=input,
                batch_s=batch_s,
                n_in=n_in,
                n_out=n_hidden[i],
                kernel=kernel)
                )
            elif i < len(n_hidden) and i > 0:
                self.hiddenLayers.append(HiddenLayer(
                rng=rng,
                input=self.hiddenLayers[i-1].output,
                batch_s=batch_s,
                n_in=n_hidden[i-1],
                n_out=n_hidden[i],
                kernel=kernel)
                )

        # The logistic regression layer gets as input the hidden units
        # of the hidden layer
        self.logRegressionLayer = LogisticRegression(
            input=self.hiddenLayers[-1].output,
            n_in=n_hidden[-1],
            n_out=n_out
        )
        # L1 norm ; one regularization option is to enforce L1 norm to
        # be small. Weights (W) mostly zero, approach the Gaussian kernels nearly to the origin.
        # The width values are not desirable to be nearly zero
        self.L1 = (
            numpy.array([abs(self.hiddenLayers[i].W).sum() for i in xrange(len(n_hidden))]).sum()
            + numpy.array([abs(self.hiddenLayers[i].s ** 2).sum() for i in xrange(len(n_hidden))]).sum()
            + abs(self.logRegressionLayer.W).sum()
        )

        # square of L2 norm ; one regularization option is to enforce
        # square of L2 norm to be small
        self.L2_sqr = (
            numpy.array([abs(self.hiddenLayers[i].W ** 2).sum() for i in xrange(len(n_hidden))]).sum()
            + numpy.array([abs(self.hiddenLayers[i].s ** 2).sum() for i in xrange(len(n_hidden))]).sum()
            + (self.logRegressionLayer.W ** 2).sum()
        )

        # negative log likelihood of the mlRKHS is given by the negative
        # log likelihood of the output of the model, computed in the
        # logistic regression layer
        self.negative_log_likelihood = (
            self.logRegressionLayer.negative_log_likelihood
        )
        # same holds for the function computing the number of errors
        self.errors = self.logRegressionLayer.errors
        # here are the predicted outputs
        self.y_pred = self.logRegressionLayer.y_pred
        # the parameters of the model are the parameters of the N layers it is
        # made out of
        self.params = []
        for i in xrange(len(n_hidden)):
            self.params += self.hiddenLayers[i].params

        self.params = self.params + self.logRegressionLayer.params
        # keep track of model input
        self.input = input


def test_mlRKHS(learning_rate=0.01, L1_reg=0.00, L2_reg=0.0001, n_epochs=1000,
             dataset='mnist.pkl.gz', batch_size=20, n_hidden=[10], verbose = False, kernel=None):
    """
    Demonstrate stochastic gradient descent optimization for a multilayer
    RKHS

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

    :type dataset: str
    :param dataset: a marker for the input files (for now, this is the
     sentence vector simensions)
   """

    dim = dataset
    missing_param = "warn"

    path = "/almac/ignacio/data/sts_all/"
    tr_px = path + "pairs-SI/vectors_H%s/pairs_eng-SI-test-2e6-nonempty_d2v_H%s_%s_m5w8.mtx" % (dim, dim, __op)
    tr_py = path + "pairs-SI/STS.gs.all-eng-SI-test-nonempty.txt"
    ts_px = path + "pairs-NO/vectors_H%s/pairs_eng-NO-test-2e6-nonempty_d2v_H%s_%s_m5w8.mtx.half0" % (dim, dim, __op)
    ts_py = path + "pairs-NO/STS.gs.all-eng-NO-test-nonempty-half0.txt"
    vl_px = path + "pairs-NO/vectors_H%s/pairs_eng-NO-test-2e6-nonempty_d2v_H%s_%s_m5w8.mtx.half1" % (dim, dim, __op)
    vl_py = path + "pairs-NO/STS.gs.all-eng-NO-test-nonempty-half1.txt"

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

    assert (train_set_x.get_value(borrow=True).shape[1] \
            == valid_set_x.get_value(borrow=True).shape[1] \
            == test_set_x.get_value(borrow=True).shape[1]) # verify dataset dimensions

    k_classes = max(train_set_y) + 1 # for 0-based class index
    #k_classes = T.max(train_set_y, axis=0).eval() + 1

    ######################
    # BUILD ACTUAL MODEL #
    ######################

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch
    x = T.matrix('x')  # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels

    rng = numpy.random.RandomState(1234)

    # construct the classifier class
    classifier = mlRKHS(
        rng=rng,
        input=x,
        batch_s=batch_size,
        n_in=in_dimensions,
        n_hidden=n_hidden,
        n_out=k_classes,
        kernel=kernel
    )

    # the cost we minimize during training is the negative log likelihood of
    # the model plus the regularization terms (L1 and L2); cost is expressed
    # here symbolically
    cost = (
        classifier.negative_log_likelihood(y)
        + L1_reg * classifier.L1
        + L2_reg * classifier.L2_sqr
    )

    # compiling a Theano function that computes the mistakes that are made
    # by the model on a minibatch
    test_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: test_set_x[index * batch_size:(index + 1) * batch_size],
            y: test_set_y[index * batch_size:(index + 1) * batch_size]
        },
        on_unused_input=missing_param
    )

    validate_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: valid_set_x[index * batch_size:(index + 1) * batch_size],
            y: valid_set_y[index * batch_size:(index + 1) * batch_size]
        },
        on_unused_input=missing_param
    )

    # compute the gradient of cost with respect to theta (sorted in params)
    # the resulting gradients will be stored in a list gparams
    gparams = [T.grad(cost, param, disconnected_inputs=missing_param) for param in classifier.params]

    updates = [
        (param, param - learning_rate * gparam)
        for param, gparam in zip(classifier.params, gparams)
    ]

    train_model = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        },
        on_unused_input=missing_param
    )
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
                    test_losses = [test_model(i) for i in range(n_test_batches)]
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
    parser = ap(description='This script trains/applies a Multi-Layer RKHS over any input dataset of numerical representations. The main aim is to determine a set of learning parameters and architecture.')
    parser.add_argument("--hidden", help="Size of the hidden layer", metavar="hidden", default=100)
    parser.add_argument("--dims", help="Size of the input layer", metavar="dims", default=2)
    parser.add_argument("--lrate", help="The learning rate", metavar="lrate", default=0.01)
    parser.add_argument("--predict", help="Predict by loading an existent mode or train a new model (specify the file name of the trained model)", metavar="predict", default=None)
    parser.add_argument("--l1_reg", help="L1 regularization parameter", metavar="l1_reg", default=0.0)
    parser.add_argument("--l2_reg", help="L2 regularization parameter", metavar="l2_reg", default=0.00010)
    parser.add_argument("--n_epochs", help="Maximum number of training epochs", metavar="n_epochs", default=1000)
    parser.add_argument("--batch", help="Size of the training mini batch", metavar="batch", default=20)
    parser.add_argument("--kernel", help="The activation RKHS function", metavar="kernel", default=None)
    parser.add_argument("--save", help="Toggles whether you want to save the learned model", action="store_true")
    args = parser.parse_args()

    if not args.predict:
        best_iter, best_validation_loss, test_score, model = test_mlRKHS(learning_rate=float(args.lrate), batch_size=20, 
                                                                      n_epochs=int(args.n_epochs), 
                                                                      n_hidden=literal_eval(args.hidden), 
                                                                      dataset=int(args.dims), verbose=True, 
                                                                      L1_reg=float(args.l1_reg), L2_reg=float(args.l2_reg),
                                                                      kernel=args.kernel )
        if args.save:
            with open("mlRKHS_STS-all_H%s_idim%s.pkl" % (args.hidden, args.dims), 'wb') as f:
                dill.dump(model, f)

        with open("mlrk.out", "a") as f:
            f.write("%f\t%i\t%f\t%d\t%s\t%s\t%s\t%s\t%s\n" % (best_validation_loss, best_iter,
                                                        test_score, int(args.dims),
                                                        args.hidden, args.lrate, args.n_epochs,
                                                        args.l1_reg, args.l2_reg))

        print("Validation score | Iteration | test performance | \
              Dims | Hidden | Learning rate | N epochs | L1 regularizer | L2 regularizer")

        print("%f\t%i\t%f\t%d\t%s\t%s\t%s\t%s\t%s\n" % (best_validation_loss, best_iter,
                                                        test_score, int(args.dims),
                                                        args.hidden, args.lrate, args.n_epochs,
                                                        args.l1_reg, args.l2_reg))
    else:
        y_pred = predict(args.predict, int(args.dims))
        for item in y_pred:
            print (item)
