from modshogun import *
import numpy as np
from numpy import loadtxt
from scipy.spatial.distance import squareform, pdist
from argparse import ArgumentParser as ap

# Use this script for estimatig the mean (median) bandwidth of your data. Give the name of the containing a vector by row.
# the the mean (median) bandwidth can be used as centre of a range of values to be tested.

def plotter(K):
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm
    from matplotlib.ticker import LinearLocator, FormatStrFormatter
    import matplotlib.pyplot as plt
    import numpy as np

    [rows, cols] = K.shape
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    X = np.arange(0, cols, 1)
    Y = np.arange(0, rows, 1)
    X, Y = np.meshgrid(X, Y)
    surf = ax.plot_surface(X, Y, K, rstride=1, cstride=1, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
    
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()

def gamma_median_heuristic(Z, num_subsample=1000):
    """
    Computes the median pairwise distance in a random sub-sample of Z.
    Returns a \gamma for k(x,y)=\exp(-\gamma ||x-y||^2), according to the median heuristc,
    i.e. it corresponds to \sigma in k(x,y)=\exp(-0.5*||x-y||^2 / \sigma^2) where
    \sigma is the median distance. \gamma = 0.5/(\sigma^2)
    """
    
    inds = np.random.permutation(len(Z))[:np.max([num_subsample, len(Z)])]
    dists = squareform(pdist(Z[inds], 'sqeuclidean'))
    median_dist = np.median(dists[dists > 0])
    mean_dist = np.mean(dists[dists > 0])
    
    sigma = np.sqrt(0.5 * median_dist)
    
    gamma = 0.5 / (sigma ** 2)
    
    return {'sigma':sigma, 'median':median_dist, 'mean':mean_dist}, gamma, Z.shape

if __name__ == "__main__":
    parser = ap(description='This script performs different tests over any input dataset of numerical representations. The main aim is to determine an estimate of a valid range of bandwidths for such a dataset, under the assumption the user wants to use any RBF-like kernel. Also it can be tested some bandwidth entered by the user. In both cases it is possible to see the resultant kernel matrices: the values and a furface plotting (when graphics are available).')    
    parser.add_argument("-f", help="Input file name (vectors)", metavar="input_file", required=True)
    parser.add_argument("-v", help="Custom value specification. If argument for -v is not the number you want to test, you can type {'sigma':sigma, 'median':median_dist, 'mean':mean_dist}. default = 'median'", metavar="custom_value", default = 'median')
    parser.add_argument("-s", help="Number of samples to be considered. Must not be greater than the available number of them.", metavar="input_file", default=1000)
    parser.add_argument("-c", help="Toggles if you want to see a combined kernel. The default is a 5-degree Polynomial kernel.", default=False, action="store_true")
    parser.add_argument("-g", help="Toggles if you have graphics for surface plotting. You will see firstly the input data, after that the obtained Gaussian kernel. If you selected seeing combined kernel, you will see also the Polynomial kernel and after that the combined kernel (Poly-Gaussian).", default=False, action="store_true")
    
    args = parser.parse_args()

    feats = loadtxt(args.f)

    [params, g, s] = gamma_median_heuristic(np.array(feats), args.s)
    
    if unicode(args.v)[0].isnumeric():
        w = float(args.v)
    else:
        w = params[args.v]
        
    features = RealFeatures(np.array(feats).T)
    
    k1 = GaussianKernel()
    k1.set_width(w)
    k1.init(features, features)
    
    print "\nRBF kernel matrix:\n\n", k1.get_kernel_matrix()
    
    if args.c:
        commb_feats = CombinedFeatures()
        commb_feats.append_feature_obj(features)
        commb_feats.append_feature_obj(features)

        k = CombinedKernel()

        k0 = PolyKernel(10,5)
        k0.init(features, features)
            
        k.append_kernel(k0)
        k.append_kernel(k1)

        print "Polynomial kernel matrix:\n\n", k0.get_kernel_matrix()

        k.init(commb_feats, commb_feats)

        print "\nCombined kernel matrix:\n\n",k.get_kernel_matrix()

    print "\n                   ==================== Input data stats =================="
    print "\nNum vectors:",features.get_num_vectors()
    if unicode(args.v)[0].isnumeric():
        print "Custom value:", float(args.v)
    print "Sigma:", params['sigma'], "\nGamma:", g 
    print "Median pairwise distance:", params['median'] 
    print "Mean pairwise distance:", params['mean']
    print "Input shape (rows, cols):", s
    print "Suggested range [0.02*mean, 12*mean]: [", 0.02*params['mean'],", ",12*params['mean'],"]\n" 

    if args.g:
        plotter(np.array(feats))
        plotter(k1.get_kernel_matrix())
        if args.c:
            plotter(k0.get_kernel_matrix())
            plotter(k.get_kernel_matrix())
