from argparse import ArgumentParser as ap
from os.path import basename, splitext, dirname
import sys

parser = ap(description='This script converts the predictions in a dictionary of estimated outputs in to ranked sentences.')
parser.add_argument("-s", help="Input file name of sentences.", metavar="sent_file", required=True)
parser.add_argument("-p", help="Regression predictions file." , metavar="predictions", required=True)
parser.add_argument("-n", help="Percentage of output sentences.", metavar="per_sents", default=25)
parser.add_argument("-d", action='store_false', help="Score order of the output summary. The default is ascendant. typing '-d' toggles descendant.", default=True)
parser.add_argument('-e', action='store_true', help='Toggles printing the estimated scores in the output file if required.')
parser.add_argument('-c', action='store_true', help='Toggles printing source information comments in the output file.')
args = parser.parse_args()

sent_file = args.s
source = basename(args.s)
pred_file = args.p
assert 1 < int(args.n) <= 100 # Valid compression percentaje?

if not args.d and not args.e:
    ops = '_d'
elif not args.d and args.e:
    ops = '_de'
    doc_index = splitext(source)[0][-2:] # 01, 02,..,10
elif args.d and args.e:
    ops = '_e'
    doc_index = splitext(source)[0][-2:] # 01, 02,..,10
else:
    ops = ''

summ_file = splitext(args.s)[0] + ops + '_%s_summ.txt' % args.n
#summ_file = dirname(args.s) + "/summs/" + splitext(source)[0] + ops + '_summ.txt'

with open(pred_file) as f:
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
        sys.stderr.write("\nThe source you specified in the input sentence file was not found in the file of results. %s" % (source))
        exit()


with open(sent_file) as f:
    sentences = map(str.strip, f.readlines())

if len(sentences) != len(predictions):
    sys.stderr.write("Length of predictions and number of sentences does not match. %s /= %s" % (len(sentences), len(predictions)))
    exit()

Ns  = int(round(len(sentences)*(float(args.n)/100.0)))

if len(sentences) < Ns or Ns <= 0:
    Ns = len(sentences)
sys.stderr.write(":>> File: %s\n :>> Document length: %d\n :>> Compression rate: %s\n :>> Taken sentences: %d\n" % (source, len(sentences), args.n, Ns))
#print ":>> Document length: %d\n" % len(sentences)
#print ":>> Compression rate: %s\n" % args.n
#print ":>> Taken sentences: %d\n" % Ns


predictions = sorted(predictions[:Ns], key = lambda tup: tup[0])

#for r in predictions: # r =  [(j, score_1),..., (J, score_N)]
with open(summ_file, 'w') as f:
    summary = []
    if args.c:
        f.write("# Source file: %s\n" % (sent_file))
        f.write("# Estimators file: %s\n" % (pred_file))
    if args.e:
        for i, p in enumerate(predictions):
            summary.append((i, p[1], sentences[p[0]]))

        for s in xrange(Ns):
            f.write("%03d\t%s\t%.4f\t%s\n" % (summary[s][0], doc_index, summary[s][1], summary[s][2]))        
    else:
        for p in predictions:
            summary.append(sentences[p[0]])
        
        for s in summary:
            f.write("%s\n" % (s))        
