import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from pdb import set_trace as st

from argparse import ArgumentParser as ap
parser = ap(description='This script trains/applies a SVR over any input dataset of numerical representations. The main aim is to determine a set of learning parameters')
parser.add_argument("-x", help="Input file name (vectors)", metavar="input_file", required=True)
parser.add_argument("-y", help="""Regression labels file. Do not specify this argument if you want to uniauely predict over any test set. In this case, you must to     
                                    specify the SVR model to be loaded as the parameter of the option -o.""", metavar="regrLabs_file", default = None)
args = parser.parse_args()

X_tr = np.loadtxt(args.x)

#st()

y = np.loadtxt(args.y)

for degree in [2, 3, 4, 5]:
    model = make_pipeline(PolynomialFeatures(degree), Ridge())
    model.fit(X_tr, y)
    y_plot = model.predict(X_tr)
    plt.plot(range(len(X_tr)), y_plot, label="degree %d" % degree)

plt.legend(loc='lower left')

plt.show()
