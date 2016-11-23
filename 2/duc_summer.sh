#!/bin/bash
shopt -s extglob

#for j in {23,25,28,48}; # index $j
for j in {23,25,28,48};
do
    for d in `ls $DATA/summer/DUC04_cluster/`; # cluster $d
    do
        for doc in `ls $DATA/summer/DUC04_cluster/"$d"/d*t-!(*summ).txt`; # document $doc
        do
            name=(`basename "$doc"`) && Dir=$DATA/summer/DUC04_cluster/"$d"/summs/LP_idx"$j"_H20;
            mkdir -p "$Dir";
            python get_summ_rank.py -ed -s "$doc" -n 5 -p "$NLP"/svr_idx"$j"_DUC04-LP-"$d"_d2v_H20_m5w8_predictions.out -m 0;
            awk '{$1=$2=$3=""; print $0}' $DATA/summer/DUC04_cluster/"$d"/"${name%.txt*}"_de_5_summ.txt > "$Dir"/"${name%.txt*}"_idx"$j"_5_NoW-LP_H20_summ.txt; 
        done;
    done;
done
