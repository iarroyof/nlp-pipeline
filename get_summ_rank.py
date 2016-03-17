from argparse import ArgumentParser as ap
from os.path import basename, splitext

parser = ap(description='This script converts the predictions in a dictionary of estimated outputs in to ranked sentences.')
parser.add_argument("-s", help="Input file name of sentences.", metavar="sent_file", required=True)
parser.add_argument("-p", help="Regression predictions file." , metavar="predictions", required=True)
parser.add_argument("-n", help="Number of output sentences.", metavar="output_length", default=8)
#parser.add_argument("-S", help="Output file name of the summary.", metavar="summ_file", required=True)
parser.add_argument('-e', action='store_true', help='Toggles printing the estimated scores in the output file if required.')
args = parser.parse_args()

sent_file = args.s
source = basename(args.s)
pred_file = args.p
summ_file = splitext(args.s)[0]+'_summ.txt'
Ns  = args.n

with open(pred_file) as f:
    predictions = []
    empty = True
    for p in f.readlines():
        s = eval(p.strip())
        if source == s['source']:
            eo = s['estimated_output']
            r = range(len(eo))
            predictions.append(sorted(zip(r, eo), reverse = True, key = lambda tup: tup[1]))
            empty = False
                  
    if empty:
        print "The source you specified in the input sentence file was not found in the file of results."
        exit()

with open(sent_file) as f:
    sentences = map(str.strip, f.readlines())

if len(sentences) != len(predictions[0]):
    print "Length of predictions and number of sentences does not match. %s /= %s" % (len(sentences), len(predictions[0][1]))
    exit()

for r in predictions: # r =  [(j, score_1),..., (J, score_N)]
    with open(summ_file, 'a') as f:
        summary = []
        if args.e:
            for p in r:
                summary.append("%.4f\t%s\n" % (p[1], sentences[p[0]]))
        else:
            for p in r:
                summary.append("%s\n" % (sentences[p[0]]))

        f.write("# Source file: %s\n" % (sent_file))
        f.write("# Estimators file: %s\n" % (pred_file))
        for n in xrange(Ns):
            f.write(summary[n])
        
