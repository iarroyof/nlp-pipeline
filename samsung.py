

path = "/almac/ignacio/data/sts_all"

train_file = '/almac/ignacio/data/sts_all/STS.input.all-eng-NO-test-nonempty.txt'
dev_file = '/almac/ignacio/data/sts_all/STS.input.all-eng-SI-test-nonempty.txt'
dev_gs_file = '/almac/ignacio/data/sts_all/pairs-SI/STS.gs.all-eng-SI-test-nonempty.txt'

# test_file = '../data/sts-en-en/manual/STS2016.manual.headlines.txt'
test_file = '/almac/ignacio/data/sts_all/sts2016-english-with-gs-v1.0/STS2016.input.plagiarism.txt'

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

train_instances = stst.load_parse_data(train_file, flag=False)
dev_instances = stst.load_parse_data(dev_file, flag=False)
model.train(train_instances, train_file)
model.test(dev_instances, dev_file)
pearsonr = stst.eval_file(model.output_file, dev_gs_file)
print(pearsonr)

test_instances= stst.load_parse_data(dev_file, server_url='http://corenlp.run')
model.test(test_instances, test_file)

