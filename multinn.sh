#!/usr/bin/env bash
#Author: multiple fora :)

# The input file (if specified) must be terminated by a empty line (the EOF in bash)
N < "$1" 

for i in {1..N}; do # load in $path a line from the input paths file
    #(python /home/iarroyof/printLine.py -l "$path") #test file
    (python nn.py /home/ignacio/data/vectors/pairs_headlines_w2v_conc_u5_70.mtx /home/ignacio/data/vectors/STS.gs.headlines_70.txt /home/ignacio/data/vectors/pairs_headlines_w2v_conc_u5_30.mtx 10) & # Execute in parallel as many (subshell) mklCalls as paths in the input file
done
