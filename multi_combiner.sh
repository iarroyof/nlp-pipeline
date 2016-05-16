
clear
Dir=$1
#model=/almac/ignacio/data/d2v_models/d2v_wikiFr_puses_H300_m10.model
model=$2
dim=$3
win=$4
outend=_H"$3"_m10w"$4".mtx
out=vectors_H"$3"

echo "Combiner for multimple sentence files contained inside a directory"
echo "Los directorios deben estar en : $Dir"

mkdir -p "$Dir"/"$out"
echo "Output directory: $Dir/$out"
for doc in `ls "$Dir"/*00.txt`;
do
    bname=`basename "$doc"`
    name="${bname%.*}"
    echo "Input filename: $name"    
    echo "Output repertoire... $Dir/$out/vectors_RPM_$name$outend"                                               #vectors_RPM_T01_C1_00_d2v_H300_m10.mtx
	python combiner.py -Si -f "$doc" -d doc2vec -w "$model" -o "$Dir"/"$out"/vectors_RPM_"$name""$outend"
done
