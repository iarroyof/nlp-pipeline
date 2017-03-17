hidd=$1
lr=$2
l1=$3
l2=$4

python $NLP/log_MLPregressor_T.py --dims 300 --hidden "$hidd" --lrate "$lr" --l1_reg "$l1" --l2_reg "$l2" --save mlp_STS-all_H300_h"$hidd".pkl
python $NLP/log_MLPregressor_T.py --dims 300 --predict mlp_STS-all_H300_h"$hidd".pkl >  mlp_STS-eval-2017_H300_h"$hidd".predict
perl $NLP/correlation-noconfidence.pl $DATA/sts_all/valid-2017/STS.gs.track5.en-en.txt mlp_STS-eval-2017_H300_h"$hidd".predict
