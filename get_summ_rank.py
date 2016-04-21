from argparse import ArgumentParser as ap
from os.path import basename, splitext, dirname
import sys
from math import modf

def Round(x):
    d, i = modf(x)
    if d > 0.5:
        return i + 1
    else:
        return i

parser = ap(description='This script converts the predictions in a dictionary of estimated outputs in to ranked sentences.')
parser.add_argument("-s", help="Input file name of sentences.", metavar="sent_file", required=True)
parser.add_argument("-p", help="Regression predictions file." , metavar="predictions", required=True)
parser.add_argument("-n", help="Percentage of output sentences.", metavar="per_sents", default=25)
parser.add_argument("-d", action='store_true', help="Score order of the output summary. The default is ascendant. typing '-d' toggles descendant.", default=False)
parser.add_argument('-e', action='store_true', help='Toggles printing the estimated scores in the output file if required.')
parser.add_argument('-c', action='store_true', help='Toggles printing source information comments in the output file.')
args = parser.parse_args()

sent_file = args.s
source = basename(args.s)
pred_file = args.p
assert 1 < int(args.n) <= 100 # Valid compression percentaje?
LME = 0 # Longueur Moyenne des l'Enonces

if not args.d and not args.e:
    ops = ''
elif not args.d and args.e:
    ops = '_e'
    doc_index = splitext(source)[0][-2:] # 01, 02,..,10
elif args.d and args.e:
    ops = '_de'
    doc_index = splitext(source)[0][-2:] # 01, 02,..,10
else:
    ops = ''

summ_file = "%s%s_%s_summ.txt" % (splitext(args.s)[0], ops, args.n) # Percentaje

with open(pred_file) as f: # open predictions file
    empty = True
    for p in f.readlines():
        s = eval(p.strip())
        if source == s['source']:
            eo = s['estimated_output']
            mxeo = max(eo); mneo = min(eo)
            r = range(len(eo))
            predictions=zip(r, eo) 
            empty = False
            break
                  
    if empty:
        sys.stderr.write("\nThe source you specified in the input sentence file was not found in the file of results. %s" % (source))
        exit()

#sys.stderr.write("\n~~~~~~~~~~~~~~~~~\n:>> Sentence scores: %s\n" % ([i[1] for i in predictions]))
sys.stderr.write("\n~~~~~~~~~~~~~~~~~\n")

with open(sent_file) as f:
    sentences = map(str.strip, f.readlines())

if len(sentences) != len(eo):
    sys.stderr.write("Length of predictions and number of sentences does not match. %s != %s" % (len(sentences), len(predictions)))
    exit()

Ns  = int(Round(len(sentences)*(float(args.n)/100.0)))

if Ns < 1:
    Ns = 1

sys.stderr.write("""\n:>> Input file: %s\n:>> Output file: %s\n:>> Document length: %d\n:>> Compression rate: %s\n:>> Taken sentences: %d\n:>> Max score: %f.3\n:>> Min score: %f.3\n""" % (source, summ_file, len(sentences), args.n, Ns, mxeo, mneo))

predictions = [(s, p) for s, p in zip(sentences, predictions) if len(s.split()) > LME]  # Filter sentences by length.
predictions=sorted(predictions, reverse = args.d, key = lambda tup: tup[1][1])
sentences = sorted(predictions[:Ns], key = lambda tup: tup[1][0]) # sort by index in origin document [(index, score),...] where score is previously sorted.
sentences, predictions = list(zip(*sentences))                    # The Ns first scores are taken, so several origin indexes will be missing.          

with open(summ_file, 'w') as f:
    summary = []
    if args.c:
        f.write("# Source file: %s\n" % (sent_file))
        f.write("# Estimators file: %s\n" % (pred_file))

    if args.e:
        for i, p in enumerate(predictions): #sentences i --> ("sentence", (doc_index, score))
            sys.stderr.write("\n:>> %d\t%f.3\t%s\n" % (i, p[1], sentences[i]))
            summary.append((i, p[1], sentences[i])) # 'i' is the index in the resulting summarized document.
        
        for s in xrange(Ns):
            f.write("%03d\t%s\t%.4f\t%s\n" % (summary[s][0], doc_index, summary[s][1], summary[s][2]))        
    else:
        for s in sentences:
            summary.append(s)
        
        for s in summary:
            f.write("%s\n" % (s))        
