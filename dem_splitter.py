from argparse import ArgumentParser as ap

parser = ap(description='This script divides a tab separated file containing a sentence and its score.')
parser.add_argument("-f", help="Input file name (complete democratic)", metavar="input_file", required=True)
parser.add_argument("-F", help="Output file name (scores complete)", metavar="output_file", required=True)

args = parser.parse_args()

with open(args.f, 'r') as fi, open(args.F, 'w') as fo:   
    for line in fi:
        fo.write("%s" % (line.split("\t")[1]))
