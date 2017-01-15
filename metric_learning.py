from metric_learn import ITML_Supervised
from data_theano import *
dim=300
__op="conc"

path = "/home/iarroyof/data/sts_all/"
tr_px = path + "pairs-SI/vectors_H%s/pairs_eng-SI-test-2e6-nonempty_d2v_H%s_%s_m5w8.mtx" % (dim, dim, __op)
tr_py = path + "pairs-SI/STS.gs.all-eng-SI-test-nonempty.txt"
ts_px = path + "pairs-NO/vectors_H%s/pairs_eng-NO-test-2e6-nonempty_d2v_H%s_%s_m5w8.mtx.half0" % (dim, dim, __op)
ts_py = path + "pairs-NO/STS.gs.all-eng-NO-test-nonempty-half0.txt"
vl_px = path + "pairs-NO/vectors_H%s/pairs_eng-NO-test-2e6-nonempty_d2v_H%s_%s_m5w8.mtx.half1" % (dim, dim, __op)
vl_py = path + "pairs-NO/STS.gs.all-eng-NO-test-nonempty-half1.txt"

datasets = load_my_data(tr_px,tr_py,ts_px,ts_py,vl_px,vl_py, shared=False)

train_set_x, train_set_y = datasets[0]
#valid_set_x, valid_set_y = datasets[1]
#test_set_x, test_set_y = datasets[2]
    
itml = ITML_Supervised(num_constraints=200)
itml.fit(train_set_x, train_set_y)
