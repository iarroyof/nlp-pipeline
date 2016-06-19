from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktParameters
from nltk.data import load
import argparse
#from pdb import set_trace as st

def add_nl(string):
    return string+"\n"


parser = argparse.ArgumentParser()
parser.add_argument("-i", help="Input file name (taw text)", metavar="input_file", required=True)
parser.add_argument("-o", help="Output file name (sentence by row)", metavar="output_file", required=True)

args = parser.parse_args()

#punkt_param = PunktParameters()
#punkt_param.abbrev_types = set(['jr', 'dr', 'vs', 'mr', 'mrs', 'prof', 'inc','e.g', 'i.e', 'gen', 'al', 'ph', 'd', 'c', 'd', 'h', 'k', 'm', 'mme', 'ms', 'msc', 'vol', 'cit', 'pp', 'ed', 'cap', 'p', 'r', 's', 'w', 'cf', 'l', 'n', 'e', 'g', 'i', 'v', 'chap', 'ch', 'i', 'no', 'st', 't', 'j', 'f', '1o', '2o', 'phd'])
#sent_tokenize = PunktSentenceTokenizer(punkt_param)
sent_tokenize = load("tokenizers/punkt/english.pickle")

with open(args.i) as f:
    lines = "".join(map(lambda str : str.replace('\t', ' ').replace('\n', ' ').replace('. »', '.»').replace('. "', '."').replace('? "', '?"').replace('? »', '?»').replace('! "', '!"').replace('! »', '!»'), f.readlines()[1:]))
#st()
with open(args.o,"w") as f:
    lines = map(add_nl, sent_tokenize.tokenize(lines))
    for sent in lines:
        f.write (sent)
