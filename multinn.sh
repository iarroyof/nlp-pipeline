#!/usr/bin/env bash
#Author: multiple fora :)

# The input file (if specified) must be terminated by a empty line (the EOF in bash)
 

for i in {1.."$1"}; do # load in $path a line from the input paths file
    python nn.py /home/ignacio/data/vectors/pairs_headlines_w2v_conc_u5_70.mtx /home/ignacio/data/vectors/STS.gs.headlines_70.txt /home/ignacio/data/vectors/pairs_headlines_w2v_conc_u5_30.mtx & 
done
