from numpy import array, loadtxt
import math
import numpy
def quantize(x):
  return array([math.floor(d) if d-int(d) < 0.5 else math.ceil(d) for d in x]).astype("int")

def load_my_data(train_x_path, train_y_path, test_x_path, test_y_path, valid_x_path, valid_y_path, shared = False):

    if shared:
        train_set_x, train_set_y = shared_dataset((loadtxt(train_x_path), quantize(loadtxt(train_y_path))))
        test_set_x, test_set_y = shared_dataset((loadtxt(test_x_path), quantize(loadtxt(test_y_path))))
        valid_set_x, valid_set_y = shared_dataset((loadtxt(valid_x_path), quantize(loadtxt(valid_y_path))))

    else:     

        train_set_x = loadtxt(train_x_path)[:40]
        train_set_y = quantize(loadtxt(train_y_path)[:40])

        test_set_x = loadtxt(test_x_path)[:10]
        test_set_y = quantize(loadtxt(test_y_path)[:10])

        valid_set_x = loadtxt(valid_x_path)[:10]
        valid_set_y = quantize(loadtxt(valid_y_path)[:10])

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]

    return rval

def shared_dataset(data_xy, borrow=True):
    import theano
    from theano import tensor as T

    data_x, data_y = data_xy
    shared_x = theano.shared(numpy.asarray(data_x, dtype=theano.config.floatX),
                                borrow=borrow)
    shared_y = theano.shared(numpy.asarray(data_y, dtype=theano.config.floatX),
                                 borrow=borrow)

    return shared_x, T.cast(shared_y, "int32")
