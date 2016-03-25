corps=(FNWN headlines OnWN)
ops=(corr sub convss conc)
Dir=/almac/ignacio/data
y=$1
for c in "${corps[@]}";
do
    for o in "${ops[@]}";
    do
       (python combiner.py -f "$Dir"/sts_"$y"/STS.input."$c".txt -d doc2vec -t "$o" -w "$Dir"/d2v_models/d2v_wikiEn_headlines_H300_m5x.model -i -o "$Dir"/pairs_"$c""$y"_d2v_H300_"$o"_m5.mtx ) &
    done
done
