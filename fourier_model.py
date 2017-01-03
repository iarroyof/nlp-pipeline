import numpy as np


# X = (x_1, x_2,.., x_m) with x_i in R^n
# Y = (y_1, y_2,.., y_m) with y_i in R

# The basis kernel
class gaussian_basis():

    def __init__(sigma, mu, x):
        self.sigma = sigma
        self.mu = mu
        self.x = x

    def compute():
        self.e_x = np.exp(-np.linalg.norm(
                               self.x - self.mu
                                         )**2/(2.*self.sigma**2)
                          )

        return self.e_x

class kernel_series():

    def __init__(basis, weights): # basis must be a list of kernel objects
        assert len(basis) == len(weights)
        self.basis = basis
        self.weights = weights
        self.size = len(weights)

    def compute():
        self.f_x = sum([w*k for w, k in zip(self.weights, self.basis)])

        return self.f_x


class fourier_classifier():

    def __init__(basis_type = "gaussian", bws = None, weights):
        self.basis_type = basis_type
        if self.basis_type == "gaussian":
            self.bws = bws
            self.weights = weights

    def compute(X):
        if self.basis_type == "gaussian":
            self.shape = X.shape
            self.basis = [array([[gaussian_basis(self.bws[i], X[i], X[j]).compute() \
                                    for i in xrange(self.shape[0])] \
                                       for j in xrange(self.shape[0]) ]
                          ) \
                            for p in xrange(len(self.weights)) ]
            self.series = kernel_series(basis = self.basis , weights = self.weights)

        return self.series.compute()

    def fit(X, y): 

    def predict(X):

        return self.compute(X)
