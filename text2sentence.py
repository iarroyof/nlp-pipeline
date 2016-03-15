import subprocess as sp

in_file = "/home/ignacio/data/Corpus_RPM2_resumes/Corpus_RPM2_documents/C1/T01_C1_03.txt"
out_file = in_file[:-4]+"-s.txt"

with open(out_file, "wb", 0) as file:
    sp.call(["perl", "/home/ignacio/data/Corpus_RPM2_resumes/split_nonum.pl", in_file], stdout=file)


