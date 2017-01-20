from w2v import *
import multiprocessing

n_cpus = 20

def mp_worker(filename):
    with open(filename) as f:
        text = f.read()
    m = clean_Ustring_fromU(text)
  
    return filename, m

def mp_handler(infiles, outfile = 'clean.txt'):
    p = multiprocessing.Pool(n_cpus)
    with open(infiles) as f:
        filenames = [line for line in (l.strip() for l in f) if line]
    with open(outfile, 'w') as f:
        for result in p.imap(mp_worker, filenames):
            # (filename, count) tuples from worker
            f.write('%s\n' % result[1])

if __name__=='__main__':

    from ast import literal_eval
    from argparse import ArgumentParser as ap

    parser = ap(description='This script cleans LATIN encoded text files from non printable chars.')
    parser.add_argument("--infiles", help="A file containing a list of input files to clean.", metavar="infiles", required=True)
    parser.add_argument("--outfile", help="A file where cleaned files must be saved together.", metavar="outfile", required=False)

    args = parser.parse_args()

    mp_handler(args.infiles, args.outfile)
