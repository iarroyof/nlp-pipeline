
clear
Dir=$1
percent=$2
#model=/home/iarroyof/data/svr_puces_model/svr_puces_complete_2_d2v_H300_esp_m10.model
preds=svr_puces_complete_2_d2v_H300_esp_m10_predictions.out

echo "Summary generation for multiple source documents and a regression predicitons file."
echo "Version 0.001 juan-manuel.torres@univ-avignon.fr"
echo "Los directorios deben estar en : $Dir"

ext=*txt
for doc in `ls $Dir$ext`;
do
    #bname=`basename $doc`
    #name="${bname%.*}"
    echo "Repertoire... $doc"
	(python get_summ_rank.py -e -s $doc -p $preds -n $percent) &
done
