from theano.tensor import _tensor_py_operators as ops
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
X = numpy.asarray(rng.uniform(low=-2.1, high=5.0, size=(batch_s, dims)))

def layer(x):

    W=theano.shared(
        value=numpy.asarray(
            rng.uniform(low=0.001, high=1.0, size=(dims, hidd_s)),
                dtype=theano.config.floatX),
        name='W', borrow=True)

    S=theano.shared(
        value=numpy.asarray(
            rng.uniform(low=10.0, high=100.0, size=(hidd_s, )),
                dtype=theano.config.floatX),
        name='S', borrow=True)

    dot_H = theano.shared(
        value=numpy.zeros((batch_s, hidd_s), 
            dtype=theano.config.floatX), 
        name='dot_H', borrow=True)
   
#    for i in range(batch_s):
#        dot_H = T.set_subtensor(dot_H[i], 
#            theano.scan(lambda w, sig:\
#                T.exp(-(w - input[i]).norm(2) ** 2 / 2 * sig ** 2),
#            sequences=[W.T, S])[0]
#            )

    for i in range(batch_s):
        for j in range(hidd_s):
            dot_H = T.set_subtensor(dot_H[i,j], 
                        T.exp(-(W.T[j] - x[i]).norm(2) ** 2) / 2 * S[j] ** 2)
       

    return dot_H, updates

layer_out = theano.function(
                            inputs=[input], 
                            outputs=layer(input),
                            on_unused_input=missing_param
                            )

print layer_out(X)