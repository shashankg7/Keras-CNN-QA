import re
import os
import numpy as np
import cPickle
import subprocess
from collections import defaultdict
import pdb, sys
from sys import stdout
from sklearn import metrics
import numpy
import json
import tqdm
from batch_generator import batch_gen
from model import load_embeddings, Model1
from alphabet import Alphabet


UNKNOWN_WORD_IDX = 0


def map_score(qids, labels, preds):
    qid2cand = defaultdict(list)
    for qid, label, pred in zip(qids, labels, preds):
      qid2cand[qid].append((pred, label))

    average_precs = []
    for qid, candidates in qid2cand.iteritems():
      average_prec = 0
      running_correct_count = 0
      for i, (score, label) in enumerate(sorted(candidates, reverse=True), 1):
        if label > 0:
          running_correct_count += 1
          average_prec += float(running_correct_count) / i
      average_precs.append(average_prec / (running_correct_count + 1e-6))
    map_score = sum(average_precs) / len(average_precs)
    return map_score


def train_model(dir):
  mode = 'TRAIN'
  if len(sys.argv) > 1:
    mode = sys.argv[1]
    if not mode in ['TRAIN', 'TRAIN-ALL']:
      print "ERROR! The two possible training settings are: ['TRAIN', 'TRAIN-ALL']"
      sys.exit(1)

  print "Running training in the {} setting".format(mode)

  data_dir = mode

  if mode in ['TRAIN-ALL']:
    q_train = numpy.load(os.path.join(data_dir, 'train-all.questions.npy'))
    a_train = numpy.load(os.path.join(data_dir, 'train-all.answers.npy'))
    y_train = numpy.load(os.path.join(data_dir, 'train-all.labels.npy'))
  else:
    q_train = numpy.load(os.path.join(data_dir, 'train.questions.npy'))
    a_train = numpy.load(os.path.join(data_dir, 'train.answers.npy'))
    y_train = numpy.load(os.path.join(data_dir, 'train.labels.npy'))

  q_dev = numpy.load(os.path.join(data_dir, 'dev.questions.npy'))
  a_dev = numpy.load(os.path.join(data_dir, 'dev.answers.npy'))
  y_dev = numpy.load(os.path.join(data_dir, 'dev.labels.npy'))
  qids_dev = numpy.load(os.path.join(data_dir, 'dev.qids.npy'))
  q_test = numpy.load(os.path.join(data_dir, 'test.questions.npy'))
  a_test = numpy.load(os.path.join(data_dir, 'test.answers.npy'))
  y_test = numpy.load(os.path.join(data_dir, 'test.labels.npy'))
  qids_test = numpy.load(os.path.join(data_dir, 'test.qids.npy'))
  f = open('vocab.json', 'r')
  vocab = json.load(f)
  max_ques_len = q_train.shape[1]
  max_ans_len = a_train.shape[1]
  embedding, dim = load_embeddings('./embeddings/embeddings.bin', vocab)
  model = Model1(dim, max_ques_len, max_ans_len, len(vocab), embedding)
  #pdb.set_trace()
  #print(model.predict([q_train, a_train]))
  # start training
  for epoch in range(8):
    for x_trainq, x_traina, y_train1 in zip(batch_gen(q_train, 50), batch_gen(a_train, 50), batch_gen(y_train, 50)):
      loss, acc = model.train_on_batch([x_trainq, x_traina], y_train1)
      perf = str(loss) + " " + str(acc)
      stdout.write("\r loss is %f with acc %f"%(loss, acc))
      stdout.flush()
      #y_train1 = y_train1.reshape(y_train1.shape[0],1)
      #x = model.predict([x_trainq, x_traina])
    y_pred = model.predict_on_batch([q_dev, a_dev])
    dev_acc = metrics.roc_auc_score(y_dev, y_pred) * 100
    test_acc = map_score(qids_dev, y_dev, y_pred) * 100
    print("dev auc is %f"%dev_acc)
    print("dev MAP is %f"%test_acc)
    
  y_pred = model.predict_on_batch([q_test, a_test])
  test_acc = metrics.roc_auc_score(y_test, y_pred) * 100
  test_map = map_score(qids_test, y_test, y_pred) * 100
  print("test auc is %f"%test_acc)
  print("test MAP is %f"%test_map)
  #model.fit([q_train, a_train], y_train, batch_size=32, nb_epoch=10, validation_data=([q_dev, a_dev], y_dev))

if __name__ == '__main__':
  # stoplist = set([line.strip() for line in open('en.txt')])
  # import string
  # punct = set(string.punctuation)
  # stoplist.update(punct)
  stoplist = None

  train = 'jacana-qa-naacl2013-data-results/train.xml'
  train_all = 'jacana-qa-naacl2013-data-results/train-all.xml'
  train_files = [train, train_all]
  train_files = [train]

  for train in train_files:
    print train

    dev = 'jacana-qa-naacl2013-data-results/dev.xml'
    test = 'jacana-qa-naacl2013-data-results/test.xml'

    train_basename = os.path.basename(train)
    name, ext = os.path.splitext(train_basename)
    outdir = '{}'.format(name.upper())
    print 'outdir', outdir

    if not os.path.exists(outdir):
      os.makedirs(outdir)

    # all_fname = train
    all_fname = "/tmp/trec-merged.txt"
    files = ' '.join([train, dev, test])
    subprocess.call("/bin/cat {} > {}".format(files, all_fname), shell=True)

    # qids, questions, answers, labels = load_data(all_fname, stoplist)
    train_model(all_fname)
    pdb.set_trace()

  
      