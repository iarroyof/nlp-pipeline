from modshogun import *
import numpy as np
from numpy import loadtxt
from scipy.spatial.distance import squareform, pdist

# Use this script for estimatig the mean (median) bandwidth of your data. Give the name of the containing a vector by row.
# the the mean (median) bandwidth can be used as centre of a range of values to be tested.

def plotter(K):
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm
    from matplotlib.ticker import LinearLocator, FormatStrFormatter
    import matplotlib.pyplot as plt
    import numpy as np
    
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    X = np.arange(0, len(K), 1)
    Y = np.arange(0, len(K), 1)
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
    print "Median dist: ", median_dist
    sigma = np.sqrt(0.5 * median_dist)
    print "Sigma: ", sigma
    gamma = 0.5 / (sigma ** 2)
    
    return sigma, gamma, median_dist, mean_dist

feats = loadtxt('vectors_w2v_puses_complete_200_m10.mtx')
#feats = loadtxt('toy_2_dim_300.data')
#feats = loadtxt('fm_ape_gutXX.txt')

[w, g, m, M] = gamma_median_heuristic(np.array(feats))

features = RealFeatures(np.array(feats).T)

commb_feats = CombinedFeatures()
commb_feats.append_feature_obj(features)
commb_feats.append_feature_obj(features)

k = CombinedKernel()

k0 = PolyKernel(10,5)
k0.init(features, features)
k1 = GaussianKernel()
k1.set_width(m)
k1.init(features, features)

k.append_kernel(k0)
k.append_kernel(k1)
#k.init(commb_feats, commb_feats)

print "Poly:\n", k0.get_kernel_matrix()
print "\nGaussian:\n", k1.get_kernel_matrix()
#print k.get_kernel_matrix()

#k.init(features, features)
k.init(commb_feats, commb_feats)

print "\nCombined:\n",k.get_kernel_matrix()

print "Num vectors: ",features.get_num_vectors()
print "Shape: ", np.array(feats).shape
print "sigma: ", w, "gamma: ", g, "median: ",m, "mean: ", M
# Comment following lines if you have not graphical interface
plotter(k0.get_kernel_matrix())
plotter(k1.get_kernel_matrix())
plotter(k.get_kernel_matrix())
