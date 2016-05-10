
clear
Dir=$1

out=tags

echo "Combiner for multimple sentence files contained inside a directory"
echo "Los directorios deben estar en : $Dir"

#mkdir -p "$Dir"/"$out"
echo "Output directory: $Dir/$out"
for i in {1..200}; do
    for doc in `ls "$Dir"/folder"$i"/lns*`; do
        bname=`basename "$doc"`
        name="${bname%.*}"
        echo "Input filename: $name"
        echo "Repertoire... $doc"
	    python3 freeling_client.py -f "$doc" > "$Dir"/folder"$i"/"$out"/"$name"_tags.txt &
    done
    wait
done
