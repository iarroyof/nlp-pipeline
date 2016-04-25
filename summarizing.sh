# We trained a SVR model over the doc2vec 300-dimmensional vectors (in np13):
# gaussTest.py -f /almac/ignacio/data/vectors_d2v_puces_complete_H300_m10.mt
# where a gamma=13.69375 was obtained
# Split a directory of documents according to 20 topics (the 2nd and 3rd characters of file names)
# Each file must contain a sentence by each row
new=$1          # Start from scratch
directory=$2    # Directrory text data is stored as a subdirectory by theme.
per=$3          # Compression rate in percentaje e.g. {5, 10, 20, 30}
dis=$4
dims=$4

d2vModel=/almac/ignacio/data/d2v_models/d2v_wikiFr_puses_H200_m10.model
svrModel=/almac/ignacio/data/svr_models/svr_puces_complete_2_d2v_H300_esp_m10.model

if [ "$new" ==  "yes" ]
then
    sure=0
    echo "Are you sure starting frm scratch??????? (yes/no)"
    read sure
    if [ "$sure" != "yes" ] 
    then
        echo "Program exit..."
        exit 111;
    fi

    cd "$directory"
    # Divide the work directory into subdirecories, according to 2 charaters from the 2nd character of each file.
    ls |  awk '{d=substr($0,2,2) ; printf "mkdir -p %s ; mv %s %s\n",d,$1,d }' | bash
#fi
    cd /almac/ignacio/nlp-pipeline
# Combiner parameters:
# model=/almac/ignacio/data/d2v_models/d2v_wikiFr_puses_H300_m10.model
# outend=_H300_m10.mtx
# out=vectors
# combiner.py -Si -f $doc -d doc2vec -w $model -o $Dir/$out/vectors_d2v_RPM_$name$outend
fi
if [ "$new" ==  "1" ] then
    for i in {01..20}; # Generate sentence vectors by inferring them from doc2vec wiki-trained model. for 20 themes.
    do                 # This procedure can take a couple of minutes by iteration. You can modify the loop control for generating vectors for two directories (topics) as maximun, due to memory constraints.
        ./multi_combiner.sh "$directory"/"$i" "$d2vModel" "$dims";
    done
fi
if [ "$new" ==  "2" ] then

    for i in {01..20}; # 20 themes.
    do
        ./multi_svr.sh "$directory"/"$i"/vectors "$svrModel"; # Generate predictions file from sentence vectors with svr.py in prediction mode.
    done                                                           # Predictions are stored in the work directory with the same name than $model plus '_predictions.out' suffix.
fi
######## Run from this line to obtain other summaries from the same dataset #########

for d in {01..20}; 
do 
    ./multi_ranker.sh "$directory"/"$d"/ "$per"; # Multiranker in descent and metadata printing mode (option -e activated)
done        # A 25% compression rate is used. This is the default. You can modify this parameter inside multi_ranker.sh

# T12_C1_e_10_summ.txt
#for i in {01..20}; # Print all by-document summaries (with printed metadata) into a single multidocument summary.
#do 
#    cat "$directory"/"$i"/T"$i"_C1_*_e_"$per"_summ* > "$directory"/"$i"/T"$i"_C1_e_"$per"_summ.txt; 
#done

for i in {01..20}; # Sort the multidocumment summary; firstly by sentence index (k1) and after by source document index (k2)
do 
    sort -k1 -n "$directory"/"$i"/T"$i"_C1_00_de_"$per"_summ.txt | sort -k2 -n > "$directory"/"$i"/T"$i"_C1_des_"$per"_summ.txt; # 's' means "sort"
done

for i in {01..20}; # Print the sorted summary without metadata (filters the first 3 columns).
do 
    awk '{$1=$2=$3=""; print $0}' "$directory"/"$i"/T"$i"_C1_des_"$per"_summ.txt > "$directory"/"$i"/T"$i"_C1_kernel_"$per"_"$dis"_summ.txt; 
done
