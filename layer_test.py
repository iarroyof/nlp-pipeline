""" (Testing) Network layer with Reproducing Kernel Hilbert Spaces (RKHS). Let x \in R^n be an input vector, w \in R^p a mean vector, s \in R^p a bandwidth vector and b \in R^p a weight vector, then the Reproducing kernel of this layer is given by:   

            f_1(x),.., f_p(x) = b_1 * k(w_1, x; s_1) + b_1,...,  b_p * k(w_p, x; s_p) + b_p
where

            k(w_j, x; s_j) = \exp(-||w_j - x||_2^2 / 2 * s_j^2)

"""

import theano.tensor as T
import numpy
import theano

theano.config.exception_verbosity="high"

batch_s=5
dims=10
hidd_s=3
out_s=2

missing_param = None #"ignore"

rng = numpy.random.RandomState(1234)
input = T.matrix("input")
X = numpy.asarray(rng.uniform(low=-2.1, high=2.0, size=(batch_s, dims)))

def layer(x):

    W=theano.shared(
        value=numpy.asarray(
            rng.uniform(low=0.001, high=1.0, size=(dims, hidd_s)),
                dtype=theano.config.floatX),
        name='W', borrow=True)

    S=theano.shared(
        value=numpy.asarray(
            rng.uniform(low=0.001, high=0.5, size=(hidd_s, )),
                dtype=theano.config.floatX),
        name='S', borrow=True)

    dot_H = theano.shared(
        value=numpy.zeros((batch_s, hidd_s), 
            dtype=theano.config.floatX), 
        name='dot_H', borrow=True)
   
    dot_H =  theano.scan(lambda x:
                theano.scan(
                    lambda w, sig: 
                        T.exp(-(w - x).norm(2) ** 2 / 2 * sig ** 2), 
                    sequences=[W.T, S])[0], 
                sequences=input
                )[0]
            
 #for i in range(batch_s):
    #    dot_H = T.set_subtensor(dot_H[i], 
    #        theano.scan(lambda w, sig:\
    #            T.exp(-(w - input[i]).norm(2) ** 2 / 2 * sig ** 2),
    #        sequences=[W.T, S])[0]
    #        )

    #for i in range(batch_s):
    #    for j in range(hidd_s):
    #        dot_H = T.set_subtensor(
    #                    dot_H[i,j], 
    #                    T.exp(
    #                        -((W.T[j] - x[i]).norm(2)) ** 2 / 2 * S[j] ** 2
    #                        )
    #                )
    return dot_H

layer_out = theano.function(
                            inputs=[input], 
                            outputs=layer(input),
                            on_unused_input=missing_param
                            )

print layer_out(X)