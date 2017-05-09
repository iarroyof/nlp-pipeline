from w2v import *
import multiprocessing
from ast import literal_eval as le
from fasttext import load_model as load_ft

n_cpus = 20


def mp_worker(filename):
    with open(filename) as f:
        text = f.read()
    utext = clean_Ustring_fromU(text)
    m = ' '.join(utext).strip()

    return m

def ln_worker(line):
    global ft

    phrase = clean_Ustring_fromU(line).replace(' ', '_')
    v=ft(phrase)
    str_v=" ".join([str(n) for n in v])
    return phrase + " " + str_v

def mp_handler(infiles=None, inlines=None, outfile = 'clean.txt'):
    p = multiprocessing.Pool(n_cpus)

    if infiles and not inlines:
        with open(infiles) as f:
            filenames = [line for line in (l.strip() for l in f) if line]
        with open(outfile, 'w') as f:
            for result in p.imap(mp_worker, filenames):
                    # (filename, count) tuples from worker
                f.write('%s\n' % result.encode('utf-8'))
    elif inlines and not infiles:
        with open(inlines) as f:
            filelines = [line for line in (l.strip() for l in f) if line]
        with open(outfile, 'w') as f:
            for result in p.imap(ln_worker, filelines):
                    # (filename, count) tuples from worker
                f.write('%s\n' % result.encode('utf-8'))

if __name__=='__main__':

    from ast import literal_eval
    from argparse import ArgumentParser as ap

    parser = ap(description='This script cleans LATIN encoded text files from non printable chars.')
    parser.add_argument("--infiles", help="A file containing a list of input files to clean.", metavar="infiles", default=None)
    parser.add_argument("--inlines", help="A file containing the input file containing lines to clean.", metavar="inlines", default=None)
    parser.add_argument("--model", help="A file containing the word embeddings bin.", metavar="model", default=None)
    parser.add_argument("--outfile", help="A file where cleaned files must be saved together.", metavar="outfile", default='clean.txt')
    args = parser.parse_args()

    global ft=load_ft(args.model)

    mp_handler(args.infiles, args.inlines, args.outfile)
