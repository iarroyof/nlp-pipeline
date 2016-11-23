# Example:

# 1.0 entity  existed in  location
# 1.0 entity  formerly existed in location
# ---------------------------------------------
# an indication that something has been present;

from pdb import set_trace as st
import numpy as np
import array
import string as strg
import argparse
from os.path import dirname, basename
exc = set(strg.punctuation.replace('-',''))

parser = argparse.ArgumentParser()
parser.add_argument("-A", help="Input file name of triplets sentence A (vectos for A and vectors-sentences B must be present in the same directory.)", metavar="input_file_tripA", required=True)
parser.add_argument("-v", action='store_true', default=False, dest = "verbose", help="Toggles if you want Verbose logs in the output.")
args = parser.parse_args()

fA = args.A                                             # input_file_triples sentence A
fa = dirname(fA) + "/" + basename(fA) + ".ftw"          # input_file_vectors sentence A
fB = dirname(fA) + "/" + "b" + basename(fA)[1:]         # input_file_triples sentence B
fb = dirname(fA) + "/" + "b" + basename(fA)[1:] + ".ftw" # input_file_vectors sentence B
comp = "00"

with open(fA, "r") as f:
    lines = f.readlines()
    from re import match
    if match("0\.\d+\\t|1\.0\\t", lines[0]): #lines[0].startswith("1.0\t"):
        trips = []
        comp = '1' + comp[1:]
        with open(fa, "r") as F:
            phr_vects = {line.strip().lower().split()[0]: np.fromstring(" ".join(line.split()[1:]), sep=" ")\
                                                                                             for line in F.readlines()} # phrases and vectors
        for line in lines:        
            trips.append(line.strip().lower().split("\t")[1:]) # all except the initial "1.0"
        try:
            terms_oieA = [{t[0].replace(' ', '_'):phr_vects[t[0].replace(' ', '_')], \
                t[1].replace(' ', '_'):phr_vects[t[1].replace(' ', '_')], \
                t[2].replace(' ', '_'): phr_vects[t[2].replace(' ', '_')]} for t in trips] # [{"nphr_a":vector, "Vphr_1":vector, "n_phr_b":vector},.., {"n_phr_a":vector, "Vphr_N":vector, "n_phr_b":vector}]
        except Exception, e: 
            print basename(fA) + ": trip :" + str(e)
            exit()
    else:
        comp = '0' + comp[1:]        
        with open(fa, "r") as F:
            lines = F.readlines()
            try:
                terms_oieA = ( lines[0].strip().lower().split()[0], np.fromstring(" ".join(lines[0].split()[1:]), sep=" ") ) # phrases and vectors
            except Exception, e: 
                print basename(fA) + ": sent :" + str(e)
                exit()

with open(fB, "r") as f:
    lines = f.readlines()
    if match("0\.\d+\\t|1\.0\\t", lines[0]):
        trips = []
        comp = comp[0] + '1'
        with open(fb, "r") as F:
            phr_vects = {line.strip().lower().split()[0]:np.fromstring(" ".join(line.split()[1:]), sep=" ") for line in F.readlines()} # phrases and vectors
        
        for line in lines:        
            trips.append(line.strip().lower().split("\t")[1:]) # all except the initial "1.0"
        try:
            terms_oieB = [{t[0].replace(' ', '_'):phr_vects[t[0].replace(' ', '_')], t[1].replace(' ', '_'):phr_vects[t[1].replace(' ', '_')], t[2].replace(' ', '_'): phr_vects[t[2].replace(' ', '_')]} for t in trips] 
        # [{"nphr_a":vector, "Vphr_1":vector, "n_phr_b":vector},.., {"n_phr_a":vector, "Vphr_N":vector, "n_phr_b":vector}]
        except Exception, e:
            print basename(fB) + ": trip :" + str(e)
            exit()
    else:
        comp = comp[0] + '0'        
        with open(fb, "r") as F:
            lines = F.readlines()
            try:
                terms_oieB = ( lines[0].strip().lower().split()[0], np.fromstring(" ".join(lines[0].split()[1:]), sep=" ") ) # phrases and vectors
            except Exception, e:
                print basename(fB) + ": sent :" + str(e)
                exit()

from scipy.spatial.distance import cosine
dist = []
trip_dist = []
if comp == "10": 
    for triplet in terms_oieA: # Fixed sentence vector B
        dist.append( (triplet.keys(), np.array([cosine(triplet[t],\
                            terms_oieB[1]) for t in triplet])) )
    trip_dist = np.mean([np.mean(d[1]) for d in dist])

elif comp == "01": 
    for triplet in terms_oieB: # Fixed sentence vector A
        dist.append( (triplet.keys(), np.array([cosine(triplet[t],\
                             terms_oieA[1]) for t in triplet])) )
    trip_dist = np.mean([np.mean(d[1]) for d in dist])

elif comp == "00": # Fixed both sentence vectors A and B 
    trip_dist = cosine(terms_oieA[1], terms_oieB[1])

elif comp == "11": # Triples both A and B
    dist = []
    for tripa in terms_oieA: 
        for tripb in terms_oieB:
            # if 
            for ta in tripa:
                for tb in tripb:
                    dist.append( (ta, tb, cosine(tripa[ta], tripb[tb])) )
    trip_dist = np.mean([np.mean(d[2]) for d in dist]) # Mean of all distances
            # else
            #dist.append(cosine(tripa[ta], tripb[tb]))

#if args.v:
#    if args.s:
#        sentences = args.s.strip().split("\t")
#        print ">> Input sentences:>> %s\n%s" % (sentences[0], sentences[1])
#    try:
#        ta = [t.keys() for t in terms_oieA]
#    except:
#        ta = terms_oieA[0]
#    try:
#        tb = [t.keys() for t in terms_oieB]
#    except:
#        tb = terms_oieB[0]
#    print ">> triplets:>> %s\n%s" % (ta, tb)
#    print ">> Distances:>> %s" % trip_dist
#    print ">> Global distance:>> %.5f" % (5.0 * trip_dist)
#else:
#    print "%.5f" % (5.0 * trip_dist)
print "%.5f" % (5.0 * trip_dist)
