# We trained a SVR model over the doc2vec 300-dimmensional vectors (in np13):
# gaussTest.py -f /almac/ignacio/data/vectors_d2v_puces_complete_H300_m10.mt
# where a gamma=13.69375 was obtained
# Split a directory of documents according to 20 topics (the 2nd and 3rd characters of file names)
# Each file must contain a sentence by each row
cd /almac/ignacio/data/summer/C1_s
ls |  awk '{d=substr($0,2,2) ; printf "mkdir -p %s ; mv %s %s\n",d,$1,d }' | bash

cd /almac/ignacio/nlp-pipeline
# Combiner parameters:
# model=/almac/ignacio/data/d2v_models/d2v_wikiFr_puses_H300_m10.model
# outend=_H300_m10.mtx
# out=vectors
# combiner.py -Si -f $doc -d doc2vec -w $model -o $Dir/$out/vectors_d2v_RPM_$name$outend
for i in {01..20}; # Generate sentence vectors by inferring them from doc2vec wiki-trained model.
do                 # This procedure can take a couple of minutes by iteration. You can modify the loop control for generating vectors for two directories (topics) as maximun, due to memory constraints.
    ./multi_combiner.sh /almac/ignacio/data/summer/C1_s/$i;
done
# model=/almac/ignacio/nlp-pipeline/pkl/svr_puces_complete_2_d2v_H300_esp_m10.model
for i in {01..20};
do
    ./multi_svr.sh /almac/ignacio/data/summer/C1_s/$i/vectors; # Generate predictions file from sentence vectors with svr.py in prediction mode.
done                                                           # Predictions are stored in the work directory with the same name than $model plus '_predictions.out' suffix.

for d in /almac/ignacio/data/summer/C1_s/*/; 
do 
    ./multi_ranker.sh "$d"; # Multiranker in descendent and metadata printing mode (option -e activated)
done                        # A 25% compression rate is used. This is the default. You can modify this parameter inside multi_ranker.sh

endf=_C1_e_summ.txt

for i in {01..20}; # Print all by-document summaries (with printed metadata) into a single multidocument summary.
do 
    cat /almac/ignacio/data/summer/C1_s/$i/T*_*summ* > /almac/ignacio/data/summer/C1_s/$i/T$i$endf; 
done

ends=_C1_e_tout_summ.txt

for i in {01..20}; # Sort the multidocumment summary; firstly by sentence index (k1) and after by source document index (k2)
do 
    sort -k1 -n /almac/ignacio/data/summer/C1_s/$i/T$i$endf | sort -k2 -n > /almac/ignacio/data/summer/C1_s/$i/T$i$ends; 
done

t=_C1_e_tout_summ.txt
x=_C1_d_tout_summ.txt

for i in {01..20}; # Print the sorted summary without metadata (filters the first 3 columns).
do 
    awk '{$1=$2=$3=""; print $0}' /almac/ignacio/data/summer/C1_s/$i/T$i$t > /almac/ignacio/data/summer/C1_s/$i/T$i$x; 
done
