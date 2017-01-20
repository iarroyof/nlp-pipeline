from argparse import ArgumentParser as ap

parser = ap(description='This script joints sentences which are in a file line each into a tab separated pair by line.')
parser.add_argument("--fin", help="Input file name, a snippet by line", metavar="fin", required=True)
parser.add_argument("--fout", help="Output file name (scores complete)", metavar="fout", required=True)
args = parser.parse_args()


with open(args.fin, "r") as f:
    lines = map(str.strip, f.readlines())
    length = len(lines)

with open(args.fout, "a") as f:
    for i in xrange(length):
        for j in xrange(length):
            f.write("%s\t%s\n" % (lines[i], lines[j]))
