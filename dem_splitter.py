
with open("/almac/ignacio/data/puces_dem.txt", 'r') as fi, open("/almac/ignacio/data/puces_scores_per.txt", 'w') as fo:   
    for line in fi:
        fo.write("%s" % (line.split("\t")[1]))
