for k in {10,20,25,35,50,100,200}
do 
    L=(`wc -l svr_all-eng-NO-test_d2v_H"$k"_esp_m5w8.out`)
    ((L--))
    for j in $(seq 0 $L)
    do 
        python svr.py -x /almac/ignacio/data/pairs_all-eng-SI-test_d2v_H"$k"_sub_m5w8.mtx -o /almac/ignacio/data/svr_models/svr_all-eng-NO-test_"$j"_d2v_H"$k"_esp_m5w8.model
    done
done
