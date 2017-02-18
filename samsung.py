

server="http://localhost:9000"

#train_file = '/almac/ignacio/data/sts_all/STS.input.all-eng-NO-test-nonempty.txt'
#dev_file = '/almac/ignacio/data/sts_all/STS.input.all-eng-SI-test-nonempty.txt'
train_file = '/almac/ignacio/data/sts_all/pairs-NO/STS.manual.all-eng-NO-test-nonempty.txt'
dev_file = '/almac/ignacio/data/sts_all/pairs-SI/STS.manual.all-eng-SI-test-nonempty.txt'
dev_gs_file = '/almac/ignacio/data/sts_all/pairs-SI/STS.gs.all-eng-SI-test-nonempty.txt'
test_file = '/almac/ignacio/data/sts_all/eval-2017/STS.manual.track5.en-en.txt'
#test_file = '/almac/ignacio/data/sts_all/sts2016-english-with-gs-v1.0/STS2016.input.plagiarism.txt'

#dev_file = '/almac/ignacio/data/sts_all/test.input.10.txt'
#train_file = dev_file#'/almac/ignacio/data/sts_all/train.input.10.txt'

#dev_gs_file = '/almac/ignacio/data/sts_all/test.gs.10.txt'
#test_file = '/almac/ignacio/data/sts_all/test.input.10.txt'

import stst


gb = stst.Classifier(stst.GradientBoostingRegression())
model = stst.Model('S1-gb', gb)

model.add(stst.LexicalFeature())
model.add(stst.ShortSentenceFeature())

model.add(stst.nGramOverlapFeature(type='lemma'))
model.add(stst.nGramOverlapFeature(type='word'))
model.add(stst.nCharGramOverlapFeature(stopwords=True))
model.add(stst.nCharGramOverlapFeature(stopwords=False))

model.add(stst.nGramOverlapBeforeStopwordsFeature(type='lemma'))
model.add(stst.nGramOverlapBeforeStopwordsFeature(type='word'))

model.add(stst.WeightednGramMatchFeature(type='lemma'))
model.add(stst.WeightednGramMatchFeature(type='word'))

model.add(stst.BOWFeature(stopwords=False))

model.add(stst.AlignmentFeature())
model.add(stst.IdfAlignmentFeature())
model.add(stst.PosAlignmentFeature())

# model.add(stst.POSFeature())

train_instances = stst.load_parse_data(train_file, flag=True, server_url=server)
#print ">>>>>>>>>>> Train shape: %s\n%d" % (train_instances, len(train_instances))
dev_instances = stst.load_parse_data(dev_file, flag=True, server_url=server)
#print ">>>>>>>>>>> Test shape: %s\n%d" % (dev_instances, len(dev_instances))
from pdb import set_trace as st
model.train(train_instances, train_file)
model.test(dev_instances, dev_file)

pearsonr = stst.eval_file(model.output_file, dev_gs_file)

print(">>>>> The Pearson's coefficient in test data is: %f.4" % pearsonr)

test_instances= stst.load_parse_data(dev_file,flag=True,  server_url=server)
model.test(test_instances, test_file)

