
clear
Dir=$1

out=tags

echo "Combiner for multimple sentence files contained inside a directory"
echo "Los directorios deben estar en : $Dir"

echo "Output directory: $Dir/$out"
for i in {01..20}; do
    for fo in `ls "$Dir"/"$i"`; do
        for doc in `ls "$Dir"/"$i"/"$fo"/*txt`; do
            bname=`basename "$doc"`
            name="${bname%.*}"
            echo "Input filename: $name"
            echo "Repertoire... $doc"
            mkdir -p "$Dir"/"$i"/"$fo"/"$out"
            python3 freeling_client.py -f "$doc" > "$Dir"/"$i"/"$fo"/"$out"/"$name"_tags.txt &
        done
    wait
    done
    #wait
done
