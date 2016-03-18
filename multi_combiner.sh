
clear
Dir=$1
model=/almac/ignacio/data/d2v_models/d2v_wikiFr_puses_H300_m10.model
#model=/home/iarroyof/w2v_models/w2v_fr_h100_m5_test.model
outend=_H300_m10.mtx
out=vectors

echo "Combiner for multimple sentence files contained inside a directory"
echo "Version 0.001 juan-manuel.torres@univ-avignon.fr"
echo "Los directorios deben estar en : $Dir"

mkdir -p $Dir/$out
echo "Output directory: $Dir/$out"
for doc in `ls $Dir/*txt`;
do
    bname=`basename $doc`
    name="${bname%.*}"
    echo "input filename: $name"    
    echo "Repertoire... $doc"
	(python combiner.py -Si -f $doc -d doc2vec -w $model -o $Dir/$out/vectors_d2v_RPM_$name$outend) &
done
