from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktParameters
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-i", help="Input file name (taw text)", metavar="input_file", required=True)
parser.add_argument("-o", help="Output file name (sentence by row)", metavar="output_file", required=True)

args = parser.parse_args()

punkt_param = PunktParameters()
punkt_param.abbrev_types = set(['dr', 'vs', 'mr', 'mrs', 'prof', 'inc','e.g', 'i.e', 'gen', 'al', 'ph', 'd', 'c', 'd', 'h', 'k', 'm', 'mme', 'ms', 'msc', 'vol', 'cit', 'pp', 'ed', 'cap', 'p', 'r', 's', 'w', 'cf', 'l', 'n', 'e', 'v', 'chap', 'ch', 'i', 'no', 'st', 't', 'j', 'f', '1o', '2o'])
sent_tokenize = PunktSentenceTokenizer(punkt_param)

with open(args.i) as f:
    lines = map(lambda str : str.replace('. »', '.»').replace('. "', '."').replace('? »', '?»').replace('! "', '!"').replace('! »', '!»'), f.readlines())

with open(args.o,"w") as f:
    for line in lines:
        sent = "\r".join(sent_tokenize.tokenize(line))
        f.write (sent)
