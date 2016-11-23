# -*- coding: utf-8 -*-
from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.cluster.hierarchy import ward, dendrogram
from sklearn.metrics.pairwise import cosine_similarity
from re import search, M, I
import logging
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import sys
import numpy as np
from argparse import ArgumentParser as ap
import os

parser = ap(description='This script trains/applies a SVR over any input dataset of numerical representations. The main aim is to determine a set of learning parameters')
parser.add_argument("-x", help="Input file name (vectors)", metavar="input_file", required=True)
parser.add_argument("-c", help="Clusterer = {km, km++, aggwr}", metavar="clusterer", required=True)
parser.add_argument("-k", help="Number of clusters", metavar="k_clusters")
parser.add_argument("-N", help="Number of trials for maximize Silhueltte", metavar="n_trials")
parser.add_argument("-t", default=False, action="store_true", help="Toggle if labels are PoS tags instead of snippets.")
parser.add_argument("-n", default=False, action="store_true", help="Toggle if labels are NounPhrases instead of snippets.")
#parser.add_argument("-y", help="""Regression labels file. Do not specify this argument if you want to uniauely predict over any test set. In this case, you must to specify
#                                the SVR model to be loaded as the parameter of the option -o.""", metavar="regrLabs_file", default = None)
args = parser.parse_args()

min_show_length = 100

def cleaner(line): # The default is the average sentence length in English
    return line.strip()#[:min_show_length]

try:
    source = search(r"(?:vectors|pairs)_([A-Za-z]+[\-A-Za-z0-9]+)_?(T[0-9]{2,3}_C[1-9]_[0-9]{2}|d\d+t|\w+)?_([d2v|w2v|fstx|coocc\w*|doc\w*]*)_(H[0-9]{1,4})_?([sub|co[nvs{0,2}|rr|nc]+]?)?_(m[0-9]{1,3}[_w?[0-9]{0,3}]?)", args.x, M|I)

    corpus = source.group(1)
    representation = source.group(3)
    dimensions = source.group(4)[1:]
    min_count = source.group(6)[1:]
    term_name = source.group(2)

except IndexError:
    print ("\nError in the filename. One or more indicators are missing. Notation: <vectors|pairs>_<source_corpus>_<model_representation>_<dimen..")
    for i in range(6):
        try:
            print (source.group(i))
        except IndexError:
            print (":>> Unparsed: %s" % (i))
            pass
    exit()
except AttributeError:
    print ("\nFatal Error in the filename. Notation: <vectors|pairs>_<source_corpus>_<model_representation>_<dimendions>_<operation>*_<mminimum_...")
    for i in range(6):
        try:
            print (source.group(i))
        except AttributeError:
            print (":>> Unparsed: %s" % (i))
            pass
    exit()

route = os.path.dirname(args.x)
## Loading files
if not args.t and not args.n:
    with open("%s/%s.txt" % (route, term_name)) as f:
        snippets = list(map(cleaner, f.readlines()))
        t = ""
elif args.n:
    with open("%s/%s.txt.phr" % (route, term_name)) as f:
        snippets = list(map(cleaner, f.readlines()))
        t = "_phr"
else:
    with open("%s/%s.tags" % (route, term_name)) as f:
        snippets = list(map(cleaner, f.readlines()))
        t = "_tags"
#TODO: Parse the snippets wit correct vectors file.
X = np.loadtxt(args.x)
if args.c.startswith("km"):
    num_clusters = int(args.k)

    if "++" in args.c:
        km = KMeans(n_clusters=num_clusters, init='k-means++', max_iter=100, n_init=1)
        km.fit(X)
        coeff = metrics.silhouette_score(X, km.labels_, sample_size=1000)
        clusters=km.labels_.tolist()
    else:
        max_silhouette = -1


        silhouette = -1
#        N_clusters = 2
        for tr in range(int(args.N)): # Number of trials for maximize Silhueltte
            
            #for N_cl in range(3,num_clusters):
             km = KMeans(n_clusters=num_clusters, init='random', max_iter=100, n_init=1)
             km.fit(X)
             coeff = metrics.silhouette_score(X, km.labels_, sample_size=1000)
             print ("Partial Silhuette: %f" % coeff)
             if silhouette < coeff: 
                 clusters=km.labels_.tolist()
                 silhouette = coeff
             #    N_clusters = N_cl
    #order_centroids = km.cluster_centers_.argsort()[:, ::-1]
    definitions=sorted(list(zip(clusters, snippets)), key=lambda x: x[0])

    while(1):
        try:
            c, s = definitions.pop()
            print ("%d\t%s" % (c, s))
        except IndexError:
            break
    
#    print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels, km.labels_))
#    print("Completeness: %0.3f" % metrics.completeness_score(labels, km.labels_))
#    print("V-measure: %0.3f" % metrics.v_measure_score(labels, km.labels_))
#    print("Adjusted Rand-Index: %.3f" % metrics.adjusted_rand_score(labels, km.labels_))
    print("Silhouette Coefficient: %0.3f" % silhouette)
    print("Number of clusters: %d" % num_clusters)
    print()

 
elif args.c.startswith("agg"):
    #if "wr" in args.c:
    dist = 1 - cosine_similarity(X)

    linkage_matrix = ward(dist) #define the linkage_matrix using ward clustering pre-computed distances

    fig, ax = plt.subplots(figsize=(15, 20)) # set size
    ax = dendrogram(linkage_matrix, orientation="right", labels=list(snippets));

    plt.tick_params(\
    axis= 'x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom='off',      # ticks along the bottom edge are off
    top='off',         # ticks along the top edge are off
    labelbottom='off')

    plt.tight_layout() #show plot with tight layout
#uncomment below to save figure
    plt.savefig("ward_clusters_%s%s_%s_H%s.png" % (term_name, t, corpus, dimensions), dpi=200) #save figure as ward_clusters
    #plt.close()
