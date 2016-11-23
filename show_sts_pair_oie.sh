stsfile=$1
nth=$2
stsdir=$(dirname $stsfile)/split_sts
#pairx=$(basename "$stsfile")

let "N=nth-1"
N=0000"$N"

echo "-------------------------------------------------"
echo "          >> Sentence pairs:"
pair=$(sed "${nth}q;d" "$stsfile")
x="\nSb -- "
echo "Sa -- $(sed 's/\t/\nSb-- /g' <(echo -e "${pair}"))"

echo "-------------------------------------------------"
echo "          >> Triplets A:"
cat "$stsdir"/a_${N:(-5)}.txt.out
echo "-------------------------------------------------"
echo "          >> Triplets B:"
cat "$stsdir"/b_${N:(-5)}.txt.out
