parallel --noswap --gnu -j12 --eta --header : 'python $NLP/log_MLRKHS.py --dims {dim} --hidden {h} --lrate {lr} --l1_reg {l1} --l2_reg {l2} --n_epochs 200 --kernel gaussian' ::: dim 300 400 ::: h [400,266,200,160] [50,10] [128,64,32,16,8] ::: lr 0.01 0.05 0.10.5 ::: l1 0.0 0.001 0.01 0.1 1.0 ::: l2 0.0 0.001 0.01 0.1 1.0
