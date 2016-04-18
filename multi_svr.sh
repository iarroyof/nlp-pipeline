clear
Dir=$1
#model=/home/iarroyof/data/svr_puces_model/svr_puces_complete_2_d2v_H300_esp_m10.model
#model=/almac/ignacio/nlp-pipeline/pkl/svr_puces_complete_2_d2v_H300_esp_m10.model
model=$2

echo "SVR for multiple vector files contained inside a directory"
echo "Version 0.001 juan-manuel.torres@univ-avignon.fr"
echo "Los directorios deben estar en : $Dir"

echo "Output directory: $Dir/$out"
for matrix in `ls "$Dir"/*v_H*mtx`;
do
    echo "Repertoire... $doc"
	(python svr.py -x "$matrix" -o "$model") &
done
