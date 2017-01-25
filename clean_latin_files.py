from w2v import *
import multiprocessing
from ast import literal_eval as le
from pdb import set_trace as st
n_cpus = 20

def mp_worker(filename):
    with open(filename) as f:
        text = f.read()
    utext = clean_Ustring_fromU(text)
    m = ' '.join(utext).strip()
  
    return m

def mp_handler(infiles, outfile = 'clean.txt'):
    p = multiprocessing.Pool(n_cpus)
    with open(infiles) as f:
        filenames = [line for line in (l.strip() for l in f) if line]
    with open(outfile, 'w') as f:
        for result in p.imap(mp_worker, filenames):
            # (filename, count) tuples from worker
            f.write('%s\n' % result.encode('utf-8'))

if __name__=='__main__':

    from ast import literal_eval
    from argparse import ArgumentParser as ap

    parser = ap(description='This script cleans LATIN encoded text files from non printable chars.')
    parser.add_argument("--infiles", help="A file containing a list of input files to clean.", metavar="infiles", required=True)
    parser.add_argument("--outfile", help="A file where cleaned files must be saved together.", metavar="outfile", default='clean.txt')

    args = parser.parse_args()

    mp_handler(args.infiles, args.outfile)
