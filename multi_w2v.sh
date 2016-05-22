# This script slides ove subdirectories of a directory ($D=/path/to/directory/{subdirectories}, subdirectories/text_file_i.txt). 
# Each slide is of 4 threads (each with its subthreads, eg, -t 8), each training the same d2v/w2v model ($model).

D=$1
model=$2
n=$3

c=0

for d in `ls "$D"`
do 
    if [ $c -lt $n ] 
    then 
        echo Current directory:"$D"/"$d"
        python w2v.py -i "$D"/"$d" -dc -t 16 -o "$model" & 
        ((c++))
    else 
        echo "-----------------------------"; 
        c=0
        wait 
    fi
done
