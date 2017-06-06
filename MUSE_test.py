import numpy as np
from utils import PAD_ID, UNK_ID, _UNK, _PAD, _END, END_ID, _BR, BR_ID, Timer, calAvgSimC, calMaxSimC
from random import sample, shuffle, random, randint
from numpy.random import choice
import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell
import sys, os
import time
import fileinput
from scipy.stats import spearmanr, pearsonr
from scipy.spatial.distance import cosine
from tensorflow.models.embedding import gen_word2vec as word2vec
from collections import defaultdict
import argparse
 

def load_arguments():
    argparser = argparse.ArgumentParser(sys.argv[0])
    argparser.add_argument("--sense_number",
            type = int,
            default = 3,
            help = "sense number per word"
        )
    argparser.add_argument("--selection_dim",
            type = int,
            default = 300,
            help = "embedding dimension for the selection module"
        )
    argparser.add_argument("--representation_dim",
            type = int,
            default = 300,
            help = "embedding dimension for the representation module"
        )
    argparser.add_argument("--dataset_dir",
            type = str,
            required = True,
            help = "directory containing the dataset"
        )
    argparser.add_argument("--ckpt_path",
            type = str,
            required = True,
            help = "path for the model checkpoint"
        )
    argparser.add_argument("--context_window",
            type = int,
            default = 5,
            help = "context window size"
        )
    argparser.add_argument("--memory",
            type = float,
            default = 0.2,
            help = "GPU memory fraction used for model training"
        )

    args = argparser.parse_args()
    return args

args = load_arguments()

batch_size = 2048
samp_size = 5
lr_rate = 0.025
lineN = 42810886

dataset_prefix = args.dataset_dir
ckptPath = args.ckpt_path
context_window = args.context_window
sense_dim = args.sense_number
embedding_dim = args.selection_dim
sense_embedding_dim = args.representation_dim

print 'selection dimension:', embedding_dim, 'representation dimension:', sense_embedding_dim
print 'sense num per word:', sense_dim

kNN = 10

word2vec_lr = 0.025

# total data in train.csv : 1194348960
vocab_size = 0
sense_size = 0

trainData = []
testData = []
testLocation = []
test_score = []
id2word = [0 for i in xrange(99160)]
word2id = dict()
id2count = [0 for i in xrange(99160)]
id2senseNum = [0 for i in xrange(99160)]
wordIDList = []
senseID = dict()

subsampling_factor = 1e-4
print_interval = 10000
max_context_length = 2*context_window + 1


trainDataSize = 863908304


with Timer('load testing data'):
  with open(dataset_prefix+'/test.sensplit.txt') as fp:
    for e, line in enumerate(fp):
      line = line.strip()
      if e % 5 == 0:
        test_score.append(float(line))
      elif e % 5 == 1 or e % 5 == 3:
        testLocation.append(int(line))
      else:
        testData.append(map(int,line.split(' ')))


with Timer('load ID data'):
  with open(dataset_prefix+'/wordID_MSSG_unigram_sensplit.txt') as fp:
    for line in fp:
      line = line.strip().split(' ')
      word = line[0]
      ID = int(line[1])
      #sense = int(line[2])
      sense = sense_dim
      freq = float(line[3])
      id2word[ID] = word
      word2id[word] = ID
      id2count[ID] = freq
      id2senseNum[ID] = sense
      #sense_dim = max(sense_dim,sense)
      vocab_size += 1
      sense_size += sense

  id2count = np.asarray(id2count)


sense_counts = [0 for i in xrange(sense_size)]
for i in xrange(vocab_size):
  for k in xrange(id2senseNum[i]):
    sense_counts[i*sense_dim+k] = id2count[i]/id2senseNum[i]

print 'vocab size =',vocab_size
print 'sense size =',sense_size

with Timer('processing test data'):
  for i in xrange(len(testData)):
    original_word = testData[i][testLocation[i]]
    original_seq = testData[i]
    if testLocation[i] - context_window < 0:
      testData[i] = [PAD_ID for t in xrange(context_window - testLocation[i])]+testData[i]
    elif testLocation[i] - context_window > 0:
      testData[i] = testData[i][testLocation[i]-context_window:]
    if len(testData[i])>2*context_window+1:
      testData[i] = testData[i][:2*context_window+1]
    elif len(testData[i])<2*context_window+1:
      testData[i] += [PAD_ID for t in xrange(2*context_window + 1 - len(testData[i]))]

    testLocation[i] = context_window
    assert(testData[i][testLocation[i]] == original_word)
    assert(len(testData[i]) == 2*context_window + 1)

  testData = np.asarray(testData)
  print testData.shape


with tf.device("/cpu:0"):
  s_out = tf.Variable(tf.zeros([sense_size, sense_embedding_dim]),\
                trainable=True, name="sense_outputs")
  s_in = tf.Variable(tf.random_uniform([sense_size, sense_embedding_dim],-(3./sense_embedding_dim)**0.5,(3./sense_embedding_dim)**0.5),\
                trainable=True, name="sense_embeddings")

  s_in_norm = tf.nn.l2_normalize(s_in, 1)

  total_words_processed = tf.Variable(0, dtype=tf.float32, trainable=False)
    #global_step = tf.Variable(0, trainable=False)
  learning_rate_word2vec = tf.Variable(float(lr_rate), trainable=False)

  words_to_train = tf.constant(float(trainDataSize))

  selected_sense_output_indices = tf.placeholder(tf.int32, None, name='selected_sense_output_indices')
  selected_sense_input_indices = tf.placeholder(tf.int32, None, name='selected_sense_input_indices')


  test_a_index = tf.placeholder(tf.int32, 1, name='test_a')
  test_b_index = tf.placeholder(tf.int32, 1, name='test_b')
  test_c_index = tf.placeholder(tf.int32, 1, name='test_c')

  test_a = tf.nn.embedding_lookup(s_in, test_a_index)
  test_b = tf.nn.embedding_lookup(s_in, test_b_index)
  test_c = tf.nn.embedding_lookup(s_in, test_c_index)


  test_question_d = tf.nn.l2_normalize(tf.add(tf.sub(test_b,test_a),test_c), dim=0)
  test_answer_d_value, test_answer_d = tf.nn.top_k(tf.matmul(test_question_d, s_in_norm, transpose_b = True), k=1, sorted=True)

  ###############################################################
  # test 0
  test_sense_indices = tf.placeholder(tf.int32, sense_dim, name='test_word_indices')
  test_word_index = tf.placeholder(tf.int32, 1, name='test_word_index')
  #s_in_l2 = tf.sqrt(tf.reduce_sum(tf.mul(s_in, s_in), 1))
  #s_in_l2 = tf.reduce_mean(tf.reshape(s_in_l2, [vocab_size, sense_dim]), 1)
  s_in_l2 = tf.reshape(s_in, [vocab_size, sense_dim, sense_embedding_dim])
  s_in_l2 = tf.transpose(s_in_l2, perm=[0,2,1])
  s_in_l2_element = tf.sub(s_in_l2, tf.batch_matmul(tf.reduce_mean(s_in_l2, 2, keep_dims=True), tf.ones([vocab_size, 1, sense_dim])))
  s_in_l2_element = tf.mul(s_in_l2_element, s_in_l2_element)
  s_in_l2_element = tf.sqrt(tf.reduce_mean(s_in_l2_element, 2))
  s_in_var = tf.reduce_mean(s_in_l2_element, 1)
  #[sense_dim, embedding_dim]
  test_input_var = tf.nn.embedding_lookup(s_in_var, test_word_index)
  embedded_test_input_norm = tf.nn.embedding_lookup(s_in_norm, test_sense_indices)
  embedded_test_input = tf.nn.embedding_lookup(s_in, test_sense_indices)
  # [sense_dim, sense_size]
  test_input_cosine = tf.matmul(embedded_test_input_norm,s_in_norm,transpose_b = True)
  test_output_logit = tf.matmul(embedded_test_input,s_out,transpose_b = True)
  #test_all_cosine = tf.matmul(embedded_test_all,n_s_all,transpose_b = True)

  test_input_l2_norm = tf.sqrt(tf.reduce_sum(tf.mul(embedded_test_input, embedded_test_input), 1))
  # = tf.s_in_l2
  test_input_topK_value, test_input_topK = tf.nn.top_k(test_input_cosine, k=(kNN+1)*sense_dim, sorted=True)
  test_output_topK_value, test_output_topK = tf.nn.top_k(test_output_logit, k=(kNN+1)*sense_dim, sorted=True)
  #test_all_topK_value, test_all_topK = tf.nn.top_k(test_all_cosine, k=kNN+1)
  ###############################################################
  # test 1
  s_in_norm = tf.reshape(s_in_norm, [vocab_size, sense_dim, sense_embedding_dim])
  s_in_norm_transpose = tf.transpose(s_in_norm, perm= [0,2,1])

  # [vocab_size, sense_dim, sense_dim]
  s_in_cosine = tf.batch_matmul(s_in_norm, s_in_norm_transpose)
  s_in_cosine = tf.reshape(s_in_cosine, [vocab_size, sense_dim*sense_dim])
  # vocab_size
  s_in_cosine_distance = tf.reduce_sum(1-tf.abs(s_in_cosine), 1)

  cosine_distance_value, cosine_distance_topK = tf.nn.top_k(s_in_cosine_distance, k=25, sorted=True)
  ###############################################################

  # [batch_size, sense_dim, sense_embedding_dim]
  embedded_sense_input = tf.nn.embedding_lookup(s_in, selected_sense_input_indices)
  embedded_sense_output = tf.nn.embedding_lookup(s_out, selected_sense_output_indices)
  embedded_sense_mix = tf.add(embedded_sense_input, embedded_sense_output) / 2
  ##############################################################

with tf.variable_scope("RLWE"):
  w_out = tf.Variable(tf.zeros([sense_size, embedding_dim]),\
                trainable=True, name="word_outputs")
  
  w_in = tf.Variable(tf.random_uniform([vocab_size, embedding_dim],-(3./embedding_dim)**0.5,(3./embedding_dim)**0.5),\
                trainable=True, name="word_embeddings")

  w_in_norm = tf.neg(tf.reduce_sum(tf.mul(w_in, w_in), 1))
  w_in_norm_topK_value, w_in_norm_topK = tf.nn.top_k(w_in_norm, k=10, sorted=True)

  global_step = tf.Variable(0, trainable=False)
  learning_rate = tf.Variable(float(lr_rate), trainable=False)
  
  context_indices = tf.placeholder(tf.int32, [context_window*2+batch_size, max_context_length])
  sense_indices = tf.placeholder(tf.int32, [(context_window*2+batch_size) * sense_dim])

  ############################
  selected_word_input_indices = tf.placeholder(tf.int32, [batch_size], name='selected_word_input_indices')
  embedded_word_input = tf.nn.embedding_lookup(w_in, selected_word_input_indices)
  ############################
  
  keep_prob = tf.placeholder(tf.float32,name='keep_probo')
  # [(context_window*2+batch_size), max_context_length, embedding_dim]
  embedded_context = tf.nn.embedding_lookup(w_in, (context_indices))
  embedded_context = tf.transpose(embedded_context, perm = [0,2,1])
  
  embedded_context = tf.reduce_sum(embedded_context,2,keep_dims=True)
  embedded_context = tf.nn.dropout(embedded_context,keep_prob=keep_prob)
  #embedded_context = tf.nn.dropout(embedded_context,keep_prob=keep_prob)
  # [(context_window*2+batch_size) * sense_dim, embedding_dim]
  embedded_word_output = tf.nn.embedding_lookup(w_out, (sense_indices))
  embedded_word_output = tf.reshape(embedded_word_output, [(context_window*2+batch_size), sense_dim, embedding_dim])

  # shape = [(context_window*2+batch_size), sense_dim, 1]
  sense_score = tf.batch_matmul(embedded_word_output, embedded_context)
  sense_score = tf.reshape(sense_score, [(context_window*2+batch_size), sense_dim])
  #
  sense_greedy = tf.to_int32(tf.argmax(sense_score, 1))
  
  # [(context_window*2+batch_size), sense_dim]
  sense_prob = tf.nn.softmax(sense_score)
  sense_maxprob = tf.reduce_max(sense_prob, 1)


saver = tf.train.Saver(tf.all_variables(), max_to_keep=None)
best_saver = tf.train.Saver(tf.all_variables(), max_to_keep=2)

def getEmbedding(sess,testData, testLocation):
  senseVec = [[] for i in xrange(len(testData))]
  senseVecOut = [[] for i in xrange(len(testData))]
  senseVecMix = [[] for i in xrange(len(testData))]
  senseScore = [[] for i in xrange(len(testData))]
  globalVec = [[] for i in xrange(len(testData))]

  currentIndex = 0
  while currentIndex<len(testData):
    #print currentIndex, len(testData)
    c_indices = []
    s_indices = []
    while (len(c_indices)!=batch_size) and currentIndex!=len(testData):
      #for i in xrange()
      c_indices.append(testData[currentIndex])
      for k in xrange(sense_dim):
        s_indices.append(testData[currentIndex][context_window]*sense_dim+k)
      currentIndex+=1
    #print currentIndex, len(testData)

    if (len(c_indices)!=batch_size) and currentIndex==len(testData):
      while (len(c_indices)!=batch_size):
        c_indices.append(c_indices[-1])
        s_indices+=[0 for i in xrange(sense_dim)]
        currentIndex+=1
    # context_window*2+batch_size
    s_indices+=[0 for i in xrange((context_window*2)*sense_dim)]
    c_indices+=[c_indices[-1] for i in xrange(context_window*2)]
    sense_probability = sess.run(sense_prob, feed_dict={context_indices: c_indices, sense_indices: s_indices, keep_prob: 1.0})

    for k in xrange(sense_dim):
      selected_s_input_indices = []
      for i in xrange(batch_size):
        if currentIndex-batch_size+i < len(testData):
          wordIndex = testData[currentIndex-batch_size+i][testLocation[currentIndex-batch_size+i]]
          selected_s_input_indices.append(wordIndex*sense_dim+ k)
        else:
          selected_s_input_indices.append(0)
      embedded_sense_in, embedded_sense_out, embedded_sense_m = sess.run([embedded_sense_input, embedded_sense_output, embedded_sense_mix], feed_dict={selected_sense_input_indices: selected_s_input_indices, selected_sense_output_indices: selected_s_input_indices})
      #print 'evaluated embedded_sense shape:', embedded_sense.shape

      for i in xrange(batch_size):
        index = currentIndex-batch_size+i
        #print index, i, batch_size, len(testData), len(senseVec), len(embedded_sense)
        if index < len(testData):
          senseVec[index].append(embedded_sense_in[i])
          senseVecOut[index].append(embedded_sense_out[i])
          senseVecMix[index].append(embedded_sense_m[i])

    selected_w_input_indices = []
    for i in xrange(batch_size):
      if currentIndex-batch_size+i < len(testData):
        wordIndex = testData[currentIndex-batch_size+i][testLocation[currentIndex-batch_size+i]]
        selected_w_input_indices.append(wordIndex)
      else:
        selected_w_input_indices.append(0)
    embedded_word = sess.run(embedded_word_input, feed_dict={selected_word_input_indices: selected_w_input_indices, keep_prob: 1.0})
    for i in xrange(batch_size):
      index = currentIndex-batch_size+i
      if index < len(testData):
        globalVec[index] = embedded_word[i]
        senseScore[index] = sense_probability[i]

  return np.asarray(senseVec), np.asarray(senseScore)

def evaluate(sess):
  with Timer("get embedding for 1st half of evaluation data"):
    [senseVec, senseScore] = getEmbedding(sess,testData, testLocation)

  indices1 = range(0,len(senseVec),2)
  indices2 = range(1,len(senseVec),2)

  avgSimC = calAvgSimC(test_score, senseVec[indices1], senseScore[indices2], senseVec[indices2], senseScore[indices2])
  maxSimC = calMaxSimC(test_score, senseVec[indices1], senseScore[indices2], senseVec[indices2], senseScore[indices2])
  
  scores = [maxSimC, avgSimC]
  score_name = ['MaxSimC', 'AvgSimC']
  
  print '============================'
  print 'AvgSimC =','{:.5f}'.format(avgSimC), 'MaxSimC =','{:.5f}'.format(maxSimC)
  print '============================'
  return [scores, score_name]

def printSenseTopK(word, topK, description):
  print description
  data = []
  usedData = set()
  for i in xrange(len(topK)):
    kNN_word = id2word[topK[i]/sense_dim]
    if kNN_word == word or kNN_word in usedData:
      continue
    data.append(id2word[topK[i]/sense_dim])
    if len(data) == kNN:
      break
  print ' '.join(data)

def printWordTopK(word, topK, description):
  print description
  printSenseTopK(word, topK[0][0:],"sense 1")
  printSenseTopK(word, topK[1][0:],"sense 2")
  printSenseTopK(word, topK[2][0:],"sense 3")
  print ''

def get_train_batch(data, leftBoundry, rightBoundry):
  c_indices = [None] * (context_window*2+batch_size)
  s_indices = [None] * ((context_window*2+batch_size)*sense_dim)

  indices = range(context_window, context_window*3+batch_size)
  for i in xrange(len(indices)):
    c_indices[i] = [PAD_ID] * (context_window - leftBoundry[indices[i]]) + data[indices[i]-leftBoundry[indices[i]]:indices[i]+1+rightBoundry[indices[i]]] + [PAD_ID] * (context_window - rightBoundry[indices[i]])
    
  for i in xrange(len(indices)):
    offset = data[indices[i]]*sense_dim
    for k in xrange(sense_dim):
      s_indices[i*sense_dim+k] = offset + k

  #annealed_keep_prob = (0.5+0.5*min(1,ite/float(total_batch)))
  input_feed = {context_indices: c_indices, sense_indices: s_indices, keep_prob: 1}
  
  return input_feed


gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.memory)
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
# Restore variables from disk.
  with Timer("Restoring model ..."):
    saver.restore(sess, ckptPath)

  scores, score_name = evaluate(sess)

  #print '\nplease enter words for kNN\n'
  while 1:
    print "\n\nPlease specify one of the following instructions (all instruction format are separated by TAB):"
    print "1. to show the k-NN for a word."
    print "format: 1\tword_token, example: 1\thead"
    print "2. to test the probabilistic policy as well as kNN for sense selection given a text context."
    print "format: 2\tsentence\tword_index, example: 2\tappoint john pope republican as head of the new army of\t5"
    print "3. to test the contexts in the training corpus, where each sense of the specified word will be selected."
    print "format: 3\tword\tprobability_threshold, example: 3\thead\t0.33"
    print "4. test the accuracy of synonym selection using the TOEFL-80 dataset (containing toefl.qst and toefl.ans)."
    print "format: 4\tdirectory, example: 4\ttoefl"
    print "5. test the accuracy of synonym selection using the ESL-50/RD-300 dataset (details refer to README)."
    print "format: 5\tfile, example: 5\tesl-rd/RD300.txt"
    print "6. dump the sense embeddings and corresponding sense indices."
    print "format: 6\tdump_embedding_file\tdump_index_file, example: 6\tembed.txt\tindex.tsv"
    line = raw_input("\nInput an instruction to conduct testing; enter <exit> to exit the program\n")
    if line == '<exit>':
      exit(0)
    line = line.strip().split('\t')

    if line[0] == '1':
      word = line[1]
      if word not in word2id:
        print 'Sorry for that "',word,'" is not in the vocabulary'
        continue
      word = word2id[word] * sense_dim
      senses = [i+word for i in xrange(sense_dim)]
      #print word/sense_dim, word2id[line[1]]
      input_topK, output_topK, input_l2_norm, input_var = sess.run([test_input_topK, test_output_topK, test_input_l2_norm, test_input_var], feed_dict={test_sense_indices:senses, test_word_index:[word/sense_dim]})
      #print 'embedding l2 norm', input_l2_norm
      #print 'average word variance per embedding dimension', input_var
      printWordTopK(id2word[word/sense_dim], output_topK, 'k-NN sorted by collocation likelihood')
    elif line[0] == '2':
      sentence = line[1].strip().lower().split(' ')
      wordIndex = int(line[2])

      if sentence[wordIndex] not in word2id:
        print '"',sentence[wordIndex], '" is not in the vocabulary'
        continue

      i = 0
      while i != len(sentence):
        if sentence[i] in word2id:
          sentence[i] = word2id[sentence[i]]
          i+=1
        else:
          if wordIndex > i:
            wordIndex -= 1
          sentence = sentence[:i]+sentence[i+1:]

      if wordIndex - context_window < 0:
        sentence = [PAD_ID for t in xrange(context_window - wordIndex)]+sentence
      elif wordIndex - context_window > 0:
        sentence = sentence[wordIndex-context_window:]
      if len(sentence)>2*context_window+1:
        sentence = sentence[:2*context_window+1]
      elif len(sentence)<2*context_window+1:
        sentence += [PAD_ID for t in xrange(2*context_window + 1 - len(sentence))]
      wordIndex = context_window

      c_indices = [None] * (context_window*2+batch_size)
      s_indices = [None] * ((context_window*2+batch_size) * sense_dim)
      for i in xrange(context_window*2+batch_size):
        c_indices[i] = sentence
        for k in xrange(sense_dim):
          s_indices[i*sense_dim+k] = sentence[wordIndex] * sense_dim + k

      sense_probability = sess.run(sense_prob, feed_dict={context_indices: c_indices, sense_indices: s_indices, keep_prob: 1.0})
      #print sense_probability[0]
      print 'probability for each sense:'
      print sense_probability[0]

      input_topK, output_topK, input_l2_norm, input_var = sess.run([test_input_topK, test_output_topK, test_input_l2_norm, test_input_var], feed_dict={test_sense_indices:s_indices[:sense_dim], test_word_index:[sentence[wordIndex]]})
      printWordTopK(id2word[sentence[wordIndex]], output_topK, 'k-NN sorted by collocation likelihood')

    elif line[0] == '3':
      word = line[1]
      threshold = float(line[2])
      if word not in word2id:
        print 'Sorry for that "',word,'" is not in the vocabulary'
        continue
      word = word2id[word] 
      senses = [i+word * sense_dim for i in xrange(sense_dim)]
      #print word/sense_dim, word2id[line[1]]
      input_topK, output_topK, input_l2_norm, input_var = sess.run([test_input_topK, test_output_topK, test_input_l2_norm, test_input_var], feed_dict={test_sense_indices:senses, test_word_index:[word/sense_dim]})
      printWordTopK(id2word[word/sense_dim], output_topK, 'k-NN sorted by collocation likelihood')


      #senseSet = set(senses)
      senseDict = dict()
      senseExample = dict()
      for s in senses:
        senseDict[s] = 5
        senseExample[s] = []

      data = [PAD_ID for i in xrange(context_window)]
      leftBoundry = [0 for i in xrange(context_window)]
      rightBoundry = [0 for i in xrange(context_window)]

      ite = 0
      getTrainBatchTime = 0.
      with open(dataset_prefix+'/train.sensplit.txt') as fp:
        for e,line in enumerate(fp):
          #print e
          if (e+1) % 5000000 == 0:
            print e+1,'/ 42810886 line, current iteration:', float(e)/42810886, getTrainBatchTime/60
            #break
          try:
            line = map(int,line.strip().split(' '))
          except:
            continue
          if word not in line:
            continue

          offset = len(data)
          counter = 0
          for i in xrange(len(line)):
            data.append(line[i])
            leftBoundry.append(min(context_window,counter))
            counter+=1
          lineLength = len(data)-offset
          for i in xrange(lineLength):
            rightBoundry.append(min(context_window,lineLength-1-i))
          #if lineLength == 1:
          #  data = data[:-1]
          #  leftBoundry = leftBoundry[:-1]
          #  rightBoundry = rightBoundry[:-1]

          assert(len(data) == len(leftBoundry))
          assert(len(data) == len(rightBoundry))

          if len(data) < batch_size+context_window*3:
            continue
          else:
            ite+=1
            #print ite

          lastTime = time.time()

          input_feed = get_train_batch(data, leftBoundry, rightBoundry)
          sense_selected, sense_max_prob = sess.run([sense_greedy, sense_maxprob], feed_dict=input_feed)
          c_indices = input_feed[context_indices]
          for i in xrange(context_window, context_window*3+batch_size):
            selected_sense = data[i] * sense_dim + sense_selected[i-context_window]
            if selected_sense in senseDict and senseDict[selected_sense] != 0 and sense_max_prob[i-context_window] > threshold:
              print '.'
              to_print = []
              for c_index in (c_indices[i-context_window]):
                if c_index != PAD_ID:
                  to_print.append(id2word[c_index])
              senseExample[selected_sense].append(' '.join(to_print))
              senseDict[selected_sense] -= 1

          if senseDict[senses[0]] == 0 and senseDict[senses[1]] == 0 and senseDict[senses[2]] == 0:
            for i in xrange(sense_dim):
              print 'example for sense', i+1
              for k in xrange(len(senseExample[word*sense_dim+i])):
                print senseExample[word*sense_dim+i][k]
              print '=================='
            break
          data = data[batch_size+context_window*2:]
          leftBoundry = leftBoundry[batch_size+context_window*2:]
          rightBoundry = rightBoundry[batch_size+context_window*2:]

          getTrainBatchTime+=(time.time()-lastTime)

        
        if len(data) < batch_size+context_window*3:
          length = len(data)
          data += [0 for k in xrange(batch_size + context_window * 3 - length)]
          leftBoundry += [0 for k in xrange(batch_size + context_window * 3 - length)]
          rightBoundry += [0 for k in xrange(batch_size + context_window * 3 - length)]
          input_feed = get_train_batch(data, leftBoundry, rightBoundry)
          sense_selected, sense_max_prob = sess.run([sense_greedy, sense_maxprob], feed_dict=input_feed)
          c_indices = input_feed[context_indices]
          for i in xrange(context_window, length):
            selected_sense = data[i] * sense_dim + sense_selected[i-context_window]
            if selected_sense in senseDict and senseDict[selected_sense] != 0 and sense_max_prob[i-context_window] > threshold:
              #print id2word[selected_sense / sense_dim], (selected_sense % sense_dim) + 1
              print (selected_sense % sense_dim) + 1, sense_max_prob[i-context_window]
              to_print = []
              for c_index in (c_indices[i-context_window]):
                if c_index != PAD_ID:
                  to_print.append(id2word[c_index])
              senseExample[selected_sense].append(' '.join(to_print))
              senseDict[selected_sense] -= 1

          for i in xrange(sense_dim):
              print 'example for sense', i+1
              for k in xrange(len(senseExample[word*sense_dim+i])):
                print senseExample[word*sense_dim+i][k]
              print '=================='

    elif line[0] == '4': # TOEFL 
      in_file_prefix = line[1]
      with Timer('testing '+in_file_prefix), open(in_file_prefix+'/toefl.qst') as qfp, open(in_file_prefix+'/toefl.ans') as afp:
        accCosine = []
        accL2 = []
        accLikelihood = []

        while True:
          line = qfp.readline()
          if not line:
            break
          words = []
          words.append(line.strip().split('\t')[-1])
          for i in xrange(4):
            line = qfp.readline()
            words.append(line.strip().split('\t')[-1])
          qfp.readline()
          ans = afp.readline().strip().split('\t')[-1]
          ans = ord(ans) - ord('a')
          afp.readline()

          available = [False for i in xrange(len(words))]
          s_indices = []
          for i in xrange(len(words)):
            if words[i] in word2id:
              words[i] = word2id[words[i]]
              available[i] = True
            else:
              words[i] = 0
            for k in xrange(sense_dim):
              s_indices.append(words[i]*sense_dim + k)

          if np.sum(available) != len(available):
            continue

          embedding_input, embedding_output = sess.run([embedded_sense_input, embedded_sense_output],{selected_sense_input_indices:s_indices, selected_sense_output_indices:s_indices})
          
          q_input = embedding_input[:sense_dim]
          q_output = embedding_output[:sense_dim]

          a_input = embedding_input[sense_dim:]
          a_output = embedding_output[sense_dim:]

          maxCosine = -1
          cosineChoice = -1

          for i in xrange(sense_dim):
            cosines = (q_input[i] / np.linalg.norm(q_input[i])) * (a_input / np.linalg.norm(a_input, axis=1)[:,np.newaxis])
            cosines = np.sum(cosines, axis = 1)
            for k in xrange(len(a_input)):
              wordIndex = k/sense_dim
              if not available[wordIndex]:
                continue
              if cosines[k] > maxCosine:
                maxCosine = cosines[k]
                cosineChoice = wordIndex

          accCosine.append(cosineChoice == ans)
          
        print 'Accuracy =', np.mean(accCosine)

    elif line[0] == '5': ## RD-300 and ESL-50
      in_file = line[1]
      with Timer('testing '+in_file), open(in_file) as fp:
        accCosine = []
              
        for line in fp:
          words = line.strip().split(' | ')
          available = [False for i in xrange(len(words))]
          s_indices = []
          for i in xrange(len(words)):
            if words[i] in word2id:
              words[i] = word2id[words[i]]
              available[i] = True
            else:
              words[i] = 0
            for k in xrange(sense_dim):
              s_indices.append(words[i]*sense_dim + k)

          if np.sum(available) != len(available):
            continue

          embedding_input, embedding_output = sess.run([embedded_sense_input, embedded_sense_output],{selected_sense_input_indices:s_indices, selected_sense_output_indices:s_indices})
          
          q_input = embedding_input[:sense_dim]
          q_output = embedding_output[:sense_dim]

          a_input = embedding_input[sense_dim:]
          a_output = embedding_output[sense_dim:]

          maxCosine = -1
          cosineChoice = -1

          for i in xrange(sense_dim):     
            cosines = (q_input[i] / np.linalg.norm(q_input[i])) * (a_input / np.linalg.norm(a_input, axis=1)[:,np.newaxis])
            cosines = np.sum(cosines, axis = 1)
            for k in xrange(len(a_input)):
              wordIndex = k/sense_dim
              if not available[wordIndex]:
                continue
              if cosines[k] > maxCosine:
                maxCosine = cosines[k]
                cosineChoice = wordIndex
          accCosine.append(cosineChoice == 0)
        print 'Accuracy =', np.mean(accCosine)

    elif line[0] == '6': ## dump all the sense vectors
    # /tmp2/models/final_established/oneside_greedy/sense_embedding.txt /tmp2/models/final_established/oneside_greedy/sense_id.tsv
      out_embedding_file = line[1]
      out_metadata_file = line[2]
      with open(out_embedding_file,'w') as embed_fp, open(out_metadata_file,'w') as meta_fp:
        for i in xrange(len(id2word)):
          if i % 1000 == 0:
            print i, len(id2word)
          embeddings = sess.run(embedded_sense_input, feed_dict={selected_sense_input_indices:range(i*sense_dim,(i+1)*sense_dim)})
          for k in xrange(sense_dim):
            meta_fp.write(id2word[i]+'_'+str(k+1)+'\n')
            embed_fp.write(' '.join(map(str,list(embeddings[k])))+'\n')
      