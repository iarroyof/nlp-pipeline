from numpy import loadtxt, savetxt
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-m", help="Input file name (sentence or pairs matrix)", metavar="input_matrix", required=True)
parser.add_argument("-f", help="Input file name (sentence or pairs text file)", metavar="input_file", required=True)
parser.add_argument("-M", help="Output file name (sentence or pairs matrix)", metavar="output_file", required=True)
parser.add_argument("-o", help="Options to add to the output file (optional, defaults to output.mtx)", default="pl", metavar="add_option")

args = parser.parse_args()

matrix = loadtxt(args.m)
with open(args.f) as f:

    for i, line in enumerate(f):
        length = len(line.split())
        