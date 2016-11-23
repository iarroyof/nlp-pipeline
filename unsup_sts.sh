stsin=$1    # Required; STS tab separated sentences file, e.g. /almac/ignacio/data/sts_13/sts13_inputs/STS.input.FNWN.txt
mod=$2      # Required; fastText model
ver=$3      # Required; options: "oie" (if compute triplet files is wanted) or "vec" (if computing vectors is wanted) or "all"
            # (compute all) or "none" (if both triplets and vectors are already computed or not wanted)
v=$4        # Optional: Toggle verbose
stsdir=$(dirname "$stsin")

. ~/.bashrc
if [ -z $ST ] || [ -z $FT ] || [ -z $DATA ]; then
    (>&2 echo "A directory is not in variables ST, FT or DATA.")
    (>&2 echo "Directories >>")
    (>&2 echo "Stanford: $ST")
    (>&2 echo "FastText: $FT")
    (>&2 echo "DATA: $DATA")
    exit 111;
fi

cpus=10 # Assign number of cpus
if [ ! -z "$v" ]; then
    if [ ! "$v" == "verbo" ]; then
        let "cpus=$v"
    fi
fi

AV=$NLP
if [ "$v" == "verbo" ]; then
    echo "Directories >>"
    echo "Stanford: $ST"
    echo "FastText: $FT"
    echo "DATA: $DATA"
    echo "Vectorizer: $AV"
fi

if [ ! -d "$stsdir"/split_sts ]; then
    mkdir "$stsdir"/split_sts
fi

i=0; cut -f 1 $stsin | while read -r l; do
    j=0000"$i"
    echo $l > "$stsdir"/split_sts/"a_${j:(-5)}.txt"
    ((i++))
done
i=0; cut -f 2 $stsin | while read -r l; do
    j=0000"$i" 
    echo $l > "$stsdir"/split_sts/"b_${j:(-5)}.txt"
    ((i++))
done
N=$(cat "$stsin" | wc -l)
Na=$(ls "$stsdir"/split_sts/a_*.txt | wc -l)
Nb=$(ls "$stsdir"/split_sts/b_*.txt | wc -l)

if [[ !( "$Na" == "$N" ) && !( "$Nb" == "$N" ) ]]; then
    (>&2 echo "Output sts line files amount does not match with input file")
    (>&2 echo "N = $N")
    (>&2 echo "Na = $Na")
    (>&2 echo "Nb = $Nb")
    exit 111
fi

Na=$(ls "$stsdir"/split_sts/a_*.txt.out | wc -l)
Nb=$(ls "$stsdir"/split_sts/b_*.txt.out | wc -l)
# Verify if files have been computed. If not do it.
if [[ !( "$Na" == "$N" ) && !( "$Nb" == "$N" ) || (("$ver" == "oie") || ("$ver" == "all")) ]]; then # If open files ie were not computed or they are wanted
    parallel --gnu -j+0 --eta --header : 'java -mx1g -cp "$ST/*" edu.stanford.nlp.naturalli.OpenIE {filea} > {filea}.out' ::: filea `ls "$stsdir"/split_sts/a_*.txt`
    parallel --gnu -j+0 --eta --header : 'java -mx1g -cp "$ST/*" edu.stanford.nlp.naturalli.OpenIE {fileb} > {fileb}.out' ::: fileb `ls "$stsdir"/split_sts/b_*.txt`
fi

if [[ ("$ver" == "vec") || ("$ver" == "all") ]]; then

    for f in `ls "$stsdir"/split_sts/*.txt.out`; do # Fill out empty triplet files with original sentence (when oie couldn't found triplets)
        if [ \! -s $f ]; then
            cat "${f%.*}" > "$f"
        fi
    done

    export mod
    export AV
    parallel --noswap --gnu -j"$cpus" --eta --header : 'bash $AV/arg22vec.sh {filea} "$mod" w' ::: filea `ls "$stsdir"/split_sts/a_*.txt.out`
    parallel --noswap --gnu -j"$cpus" --eta --header : 'bash $AV/arg22vec.sh {fileb} "$mod" w' ::: fileb `ls "$stsdir"/split_sts/b_*.txt.out`
fi

if [ "$v" == "verbo" ]; then
    export AV
    parallel -k --noswap --gnu -j+0 --eta --header : 'python $AV/trip_comp.py -A {filea}' ::: filea `ls "$stsdir"/split_sts/a_*.txt.out`
else
    export AV
    parallel -k --noswap --gnu -j+0 --eta --header : 'python $AV/trip_comp.py -A {filea} -v' ::: filea `ls "$stsdir"/split_sts/a_*.txt.out`
fi

(>&2 echo "C'est fini.")
