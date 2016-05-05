from numpy import loadtxt
from scipy.signal import correlate
from scipy.stats import pearsonr
from matplotlib.pyplot import *
from matplotlib import pyplot as pp
import matplotlib.pyplot as plt
from ast import literal_eval
from argparse import ArgumentParser
import subprocess
from os import remove
#titlle = "Prediction for sentence semantic similarity regression"
titlle = "Sentence summary candidature"
#yylabel = "Similarity score"
yylabel = "Average candidature score"
def read_results(file_name):
    out = open(file_name, 'r').readlines()
    outs = []

    for res in out:
        if res.startswith('{'):
            outs.append(literal_eval(res.strip()))
        else:
            continue
    if not outs: 
        print "not found or unparsed results"
        exit()
    return outs

def plotter(goldStandard_file, predictions_file, log_scale=False, number_result=None, same_figure=True):
    label_file = goldStandard_file #"/home/iarroyof/data/sts_test_13/STS.output.FNWN.txt"
    output_file = predictions_file # "/home/iarroyof/data/output_1_sub_ccbsp_topic.txt"

    labs = goldStandard_file
    sample = range(0, len(labs))
    est_outs = []       
    performs = []
    for est in output_file:
        est_outs.append(est['estimated_output'])
        try:
            performs.append(est['best_score'])
        except KeyError:
            try:
                performs.append(est['performance'])
            except KeyError:
                pass

    if len(labs) != len(est_outs[0]):
        print "Compared predicitons and goldStandard are not of the same length"
        print "len gs: ", len(labs), " vs len outs: ",  len(est_outs[0])
        exit()
    
    labels = sorted(zip(labs, sample), key = lambda tup: tup[0])

    ordd_est_outs = []
    true = []
    est_out = []
    ccorrs = []
 
    true = zip(*labels)[0]
    for out in est_outs:
        for i in labels:
            est_out.append(out[i[1]])
        ccorrs.append(correlate(true, est_out, mode = 'same')/len(labs))
        ordd_est_outs.append(est_out)
        est_out = []   
#set_trace()
    i = 0

    if number_result:
        grid(True)
        title(titlle+" ["+str(i+1)+"]")
        grid(True)
        p1 = Rectangle((0, 0), 1, 1, fc="r")
        p2 = Rectangle((0, 0), 1, 1, fc="b")
        p3 = Rectangle((0, 0), 1, 1, fc="g")
        legend((p1, p2, p3), ["Gd_std sorted relationship", "Predicted sorted output", "Cross correlation"], loc=4)
        xlabel('GoldStandad-sorted samples')
        ylabel(yylabel)
        if log_scale:
            yscale('log')
        
        plot(sample, true, color = 'r', linewidth=2)
        plot(sample, ordd_est_outs[number_result], color = 'b', linewidth=2)
        plot(sample, ccorrs[number_result], color = 'g', linewidth=2)
        show()
    else:	
        for est_o in ordd_est_outs:
            figure()
            grid(True)
            title(titlle+"["+str(i+1)+"]")
            grid(True)
            p1 = Rectangle((0, 0), 1, 1, fc="r")
            p2 = Rectangle((0, 0), 1, 1, fc="b")
            p3 = Rectangle((0, 0), 1, 1, fc="g")
            legend((p1, p2, p3), ["Gd_std sorted relationship", "Predicted sorted output", "Cross correlation"], loc=4)
            xlabel('GoldStandad sorted samples')
            ylabel(yylabel)
            if log_scale:
                yscale('log')
            plot(sample, true, color = 'r', linewidth=2)
            plot(sample, est_o, color = 'b', linewidth=2)
            plot(sample, ccorrs[i], color = 'g', linewidth=2)
            i += 1
            if same_figure:
                show()
       

        if not same_figure:
            show()

def pearsons(gl, el):
    with open("GL.txt", "w") as f:
        for p in gl:
            f.write("%s\n" % p)
    with open("EL.txt", "w") as f:
        for p in el:
            f.write("%s\n" % p)
    gs="GL.txt"
    est="EL.txt"
    
    pipe = subprocess.Popen(["perl", "./correlation-noconfidence.pl", gs, est], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    pipe.stdin.write("%s %s" % (gs, est))
    try:
        pearson= float(str(pipe.stdout.read()).split()[1])
    except:
        print str(pipe.stdout.read())
        exit()

    remove(gs)
    remove(est)
    pipe.stdin.close()
    pipe.stdout.close()
    
    return pearson

parsed = ArgumentParser(description='Plots desired labels, predicted outputs and calculates their corss-correlation.')
parsed.add_argument('-g', type=str, dest = 'goldStandard_file', help='Specifies the goldStandard file.')
parsed.add_argument('-p', type=str, dest = 'predictions_file', help='Specifies the machine predictions file.')
parsed.add_argument('-l', action='store_true', dest = 'log_scale', help='Toggles log scale for plotting.')
parsed.add_argument('-r', type=int, dest = 'number_result', help='If you know wath of all input results only to show, give it.')
parsed.add_argument('-s', action='store_true', dest = 'same_figure', help='Toggles plotting all loaded results in the same figure or each result in a different figure.')
parsed.add_argument('-S', action='store_true', dest = 'Sorted_outs', help='Toggles plotting sorted output and goldstandard.')
args = parsed.parse_args()
label_file = args.goldStandard_file #"/home/iarroyof/data/sts_test_13/STS.output.FNWN.txt"
output_file = args.predictions_file # "/home/iarroyof/data/output_1_sub_ccbsp_topic.txt"

from pdb import set_trace as st
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score
labs = loadtxt(label_file)
sample = range(0, len(labs))
est_outs = []       
models = []
weights = []
performs = []
params = []

results = read_results(output_file)
if not args.number_result:
    results = sorted(results, key = lambda k: k['performance'], reverse = True)

for est in results:
    est_outs.append(est['estimated_output'])
    try:
        models.append(est['learned_model'])
        #if 'file' in est['learned_model']:
        #    model = False # Es solo en tanto no se decodifique aqui el modelo desde un pickle (svr)
    except KeyError:
        model = False
        pass

    try:
        performs.append(est['best_score'])
    except KeyError:
        try:
            performs.append(est['performance'])
        except KeyError:
            performs.append(0.0)
            pass 
    try:   
        params.append(est['best_params'])
    except:
        params.append("none")
        pass 
        
for out in est_outs:
    if len(labs) != len(out):
        print "Compared predicitons and goldStandard are not of the same length"
        print "len gs: ", len(labs), " vs len outs: ",  len(est_outs[0])
        exit()
    
labels = sorted(zip(labs, sample), key = lambda tup: tup[0])

ordd_est_outs = []
true = []
est_out = []
ccorrs = []
 
true = zip(*labels)[0]
for out in est_outs:
    for i in labels:
        est_out.append(out[i[1]])
    ccorrs.append(correlate(true, est_out, mode = 'same')/len(labs))
    ordd_est_outs.append(est_out)
    est_out = []   

i = 0

if args.number_result:
    
    try:
        x = range(len(models[args.number_result]['weights'])); 
        y = models[args.number_result]['weights']    
        f, axarr = plt.subplots(1, 3)
        model = True
    except KeyError:
        f, axarr = plt.subplots(1, 1)
        model = False
    except IndexError:
        f, axarr = plt.subplots(1, 1)
        model = False
    
    MSE = mse(labs, est_outs[int(args.number_result)])
    print  "%d:" % args.number_result, params[int(args.number_result)], "perform: %f, %f" % (performs[int(args.number_result)], MSE)
    print models[int(args.number_result)],"\n"
    #pearson = pearsonr(true, ordd_est_outs[args.number_result])
    r2 = r2_score(labs, est_outs[int(args.number_result)])
    pearson = pearsons(labs, est_outs[int(args.number_result)])
    grid(True)
    title( "%s [%d],\nPearson: %.5f, perform: %.4f" % (titlle, args.number_result, pearson, r2)) #performs[args.number_result]))
    grid(True)
    p1 = Rectangle((0, 0), 1, 1, fc="r")
    p2 = Rectangle((0, 0), 1, 1, fc="b")
    #p3 = Rectangle((0, 0), 1, 1, fc="g")
    #axarr[0].legend((p1, p2, p3), ["Gd_std sorted relationship", "Predicted sorted output", "Cross correlation"], loc=4)
    axarr.legend((p1, p2), ["Gd_std sorted relationship", "Regression estimation"], loc='best')
    axarr.set_xlabel('GoldStandad sorted samples')
    axarr.set_ylabel(yylabel)
    if args.log_scale and model:
        axarr.set_yscale('log')
        #axarr.set_yscale('log')
    
    axarr.plot(sample, true, color = 'r', linewidth=2)
    axarr.plot(sample, ordd_est_outs[args.number_result], color = 'b', linewidth=2)
    #axarr[0].plot(sample, ccorrs[args.number_result], color = 'g', linewidth=2)
    #axarr.set_title('Predictions, goldstandard and cross correlation')
    if model:
        axarr[1].scatter(x, y)
        axarr[1].set_title('Learned weights')
        y = models[args.number_result]['widths']
        axarr[2].scatter(x, y)
        axarr[2].set_title('Generated widths')    
        axarr[2].set_xlabel('Basis '+models[args.number_result]['family']+' index')

    show()
else:	
    k = 0
    for est_o in ordd_est_outs:
        figure()
        grid(True)
        pearson = pearsons(labs, est_outs[k])
        MSE = mse(labs, est_outs[k])
        #pearson = pearsonr(labs, est_outs[k])[1]
        k += 1
        print  "%d:" % k, params[k-1], "perform: %f, %f" % (performs[k-1], MSE)
        print models[k-1],"\n"
        title( "%s [%d],\npearson: %.5f, perform: %.4f" % (titlle, k, pearson, performs[k-1]) )
        grid(True)
        p1 = Rectangle((0, 0), 1, 1, fc="r")
        p2 = Rectangle((0, 0), 1, 1, fc="b")
        
        #p3 = Rectangle((0, 0), 1, 1, fc="g")
        #p4 = Rectangle((0, 0), 1, 1, fc="b")
        #legend((p1, p2, p3, p4), ["Gd_std sorted relationship", "Predicted sorted output", "Goldstandard", "Prediction"], loc="best")
        legend((p1, p2), ["Goldstandard relationship", "Regression estimation"], loc='best')
        xlabel('Samples')
        ylabel(yylabel)
        if args.log_scale:
            yscale('log')
        if args.Sorted_outs:
            plot(sample, true, color = 'r', linewidth=2)
            plot(sample, est_o, color = 'b', linewidth=2)
        else:
            plot(sample, labs, color = 'r', linewidth=2)
            plot(sample, est_outs[k-1], color = 'b', linewidth=2)
        i += 1
        if args.same_figure:
            show()
       

    if not args.same_figure:
        show()

