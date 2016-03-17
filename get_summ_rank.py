from argparse import ArgumentParser as ap
from os.path import basename, splitext

parser = ap(description='This script converts the predictions in a dictionary of estimated outputs in to ranked sentences.')
parser.add_argument("-s", help="Input file name of sentences.", metavar="sent_file", required=True)
parser.add_argument("-p", help="Regression predictions file." , metavar="predictions", required=True)
parser.add_argument("-n", help="Number of output sentences.", metavar="num_sents", default=8)
parser.add_argument("-d", action='store_false', help="Score order of the output summary. The default is ascendant. typing '-d' toggles descendant.", default=True)
parser.add_argument('-e', action='store_true', help='Toggles printing the estimated scores in the output file if required.')
args = parser.parse_args()

sent_file = args.s
source = basename(args.s)
pred_file = args.p
summ_file = splitext(args.s)[0]+'_summ.txt'
Ns  = args.n
from pdb import set_trace as st
with open(pred_file) as f:
    predictions = []
    empty = True
    for p in f.readlines():
        s = eval(p.strip())
        if source == s['source']:
            eo = s['estimated_output']
            r = range(len(eo))
            predictions=sorted(zip(r, eo), reverse = args.d, key = lambda tup: tup[1])
            empty = False
            break
                  
    if empty:
        print "The source you specified in the input sentence file was not found in the file of results. %s" % (source)
        exit()

with open(sent_file) as f:
    sentences = map(str.strip, f.readlines())

if len(sentences) != len(predictions):
    print "Length of predictions and number of sentences does not match. %s /= %s" % (len(sentences), len(predictions))
    exit()
st()
predictions = sorted(predictions[:Ns], key = lambda tup: tup[0])

#for r in predictions: # r =  [(j, score_1),..., (J, score_N)]
with open(summ_file, 'a') as f:
    summary = []
    f.write("# Source file: %s\n" % (sent_file))
    f.write("# Estimators file: %s\n" % (pred_file))
    if args.e:
        for p in predictions:
            summary.append((p[1], sentences[p[0]]))
        #summary = sorted(summary, key=lambda tup:tup[1][0])
        for s in xrange(Ns):
            f.write("%.4f\t%s\n" % (summary[s][0], summary[s][1]))        
    else:
        for p in r:
            summary.append(sentences[p[0]])
        #summary = sorted(summary, key=lambda tup:tup[0])
        for s in xrange(Ns):
            f.write("%s\n" % (summary[s][1]))        
