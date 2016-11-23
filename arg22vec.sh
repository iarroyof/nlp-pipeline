arg2=$1
mod=$2
wrd=$3
cat "$arg2" | tr " " "_" | sed 's/____//' > "$arg2".phr
if [ ! -z "$mod" ] && [ "${mod##*.}" == "bin" ]; then
    if [ -z "$arg2".phr ] || [ -z "$arg2" ]; then
        echo "No file found"
        echo "$arg"
        exit 111
    fi
    sed -i '/^$/d' "$arg2".phr
    sed -i '/^$/d' "$arg2"
    if [ -z "$wrd" ]; then
        "$FT"/fasttext print-vectors "$mod" < "$arg2".phr | awk -F ' ' '{$1=""; print $0}' > "$arg2".ft
    else
        "$FT"/fasttext print-vectors "$mod" < "$arg2".phr > "$arg2".ftw
    fi
else
    echo "No model was specified..."
    echo "Model name entered: $mod"
    exit 111
fi
