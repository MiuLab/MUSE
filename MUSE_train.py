import numpy as np
from utils import PAD_ID, UNK_ID, _UNK, _PAD, _END, END_ID, _BR, BR_ID, Timer, calAvgSimC, calMaxSimC
from random import sample, shuffle, random, randint
from numpy.random import choice
import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell
import sys, os
import time
import fileinput
from tensorflow.models.embedding import gen_word2vec as word2vec
import argparse

def load_arguments():
    argparser = argparse.ArgumentParser(sys.argv[0])
    argparser.add_argument("--formulation",
            type = str,
            default = 'one-sided_optimization',
            help = "original_formulation/one-sided_optimization/value_function_factorization"
        )
    argparser.add_argument("--learning_method",
            type = str,
            default = 'Q-greedy',
            help = "Q-greedy/Q-epsilon-greedy/Q-Boltzmann/policy/smooth_policy"
        )
    argparser.add_argument("--lr_rate",
            type = float,
            default = 0.025,
            help = "learning rate"
        )
    argparser.add_argument("--batch_size",
            type = int,
            default = 2048,
            help = "batch size"
        )
    argparser.add_argument("--sample_size",
            type = int,
            default = 25,
            help = "number of negative samples for skip-gram"
        )
    argparser.add_argument("--context_window",
            type = int,
            default = 5,
            help = "context window size"
        )
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
    argparser.add_argument("--save_dir",
            type = str,
            required = True,
            help = "directory for saving checkpoints (warning: lots of space will be consumed)"
        )
    argparser.add_argument("--log_path",
            type = str,
            required = True,
            help = "path for saving training logs"
        )
    argparser.add_argument("--memory",
            type = float,
            default = 0.2,
            help = "GPU memory fraction used for model training"
        )
    argparser.add_argument("--ckpt_threshold",
            type = float,
            default = 0.6,
            help = "start checkpoint saving for best models (MaxSimC and AvgSimC) after the threshold is reached."
        )
    args = argparser.parse_args()
    return args

args = load_arguments()

lineN = 42810886

formulation = args.formulation
learning_method = args.learning_method
lr_rate = args.lr_rate
batch_size = args.batch_size
samp_size = args.sample_size
context_window = args.context_window
sense_dim = args.sense_number
embedding_dim = args.selection_dim
sense_embedding_dim = args.representation_dim
dataset_prefix = args.dataset_dir
save_dir = args.save_dir
log_path = args.log_path

if not os.path.exists(save_dir):
  print save_dir, 'not exist!!'
  exit(0)
else:
  if save_dir[-1] != '/':
    print 'please enter a directory for save_dir (path ending with /)'
    exit(0)
  print save_dir, 'exist!\nckpt storage is ok'

learning_framework = None
if learning_method in set(['Q-epsilon-greedy', 'Q-greedy', 'Q-Boltzmann']):
  learning_framework = 'value_based'
elif learning_method in set(['policy', 'smooth_policy']):
  learning_framework = 'policy_gradient'
else:
  print learning_method, 'is not supported'
  exit(0)

if formulation not in set(['original_formulation', 'one-sided_optimization', 'value_function_factorization']):
  print formulation, 'is not supported'
  exit(0)

if learning_framework == 'policy_gradient' and formulation == 'value_function_factorization':
  print 'policy_gradient and value_function_factorization are not compatible'
  exit(0)

print 'learning method:', learning_method
print 'learning_framework:', learning_framework
print 'formulation:', formulation

print 'lr_rate',lr_rate, 'batch size=', batch_size, 'sample size', samp_size, 'context_window', context_window, 'sense dim', sense_dim


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
senseID = dict()

subsampling_factor = 1e-4

print_interval = 10000
training_epochs = 2000

max_context_length = 2*context_window + 1

print 'sense embedding dimension =', sense_embedding_dim
print 'embedding dimension =', embedding_dim

trainDataSize = 863908304
print 'trainDataSize',trainDataSize

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
      sense = sense_dim
      freq = float(line[3])
      id2word[ID] = word
      word2id[word] = ID
      id2count[ID] = freq
      id2senseNum[ID] = sense
      vocab_size += 1
      sense_size += sense

  id2count = np.asarray(id2count)
  id2freq = id2count / np.sum(id2count)

rnd = []
for i in xrange(2048):
  rnd.append(random())
for i in xrange(len(id2freq)):
  if i <= 3:
    continue
  if id2freq[i] == 0:
    continue
  id2freq[i] = (np.sqrt(id2freq[i]/subsampling_factor)+1) * (subsampling_factor/id2freq[i])

initialWordIn = [[] for i in xrange(vocab_size)]
initialSenseIn = [[] for i in xrange(sense_size)]
initialWordOut = [[] for i in xrange(sense_size)]

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

  total_words_processed = tf.Variable(0, dtype=tf.float32, trainable=False)
    #global_step = tf.Variable(0, trainable=False)
  learning_rate_word2vec = tf.Variable(float(lr_rate), trainable=False)

  selected_sense_output_indices = tf.placeholder(tf.int32, None, name='selected_sense_output_indices')
  selected_sense_input_indices = tf.placeholder(tf.int32, None, name='selected_sense_input_indices')

  # [batch_size, sense_dim, sense_embedding_dim]
  embedded_sense_input = tf.nn.embedding_lookup(s_in, selected_sense_input_indices)
  embedded_sense_output = tf.nn.embedding_lookup(s_out, selected_sense_output_indices)

  reward_sense_prob = tf.sigmoid(tf.reduce_sum(tf.mul(embedded_sense_input, embedded_sense_output), 1))
  print 'embedded_sense_input:shape=',embedded_sense_input

  inc = total_words_processed.assign_add(batch_size)
  with tf.control_dependencies([inc]):
    train = word2vec.neg_train(s_in,s_out,selected_sense_input_indices,selected_sense_output_indices,\
      learning_rate_word2vec,vocab_count=sense_counts,num_negative_samples=samp_size)
  
  init_word2vec = tf.initialize_all_variables()

with tf.variable_scope("RLWE"):
  w_out = tf.Variable(tf.zeros([sense_size, embedding_dim]),\
                trainable=True, name="word_outputs")
  
  w_in = tf.Variable(tf.random_uniform([vocab_size, embedding_dim],-(3./embedding_dim)**0.5,(3./embedding_dim)**0.5),\
                trainable=True, name="word_embeddings")

  global_step = tf.Variable(0, trainable=False)
  learning_rate = tf.Variable(float(lr_rate), trainable=False)
  
  context_indices = tf.placeholder(tf.int32, [context_window*2+batch_size, max_context_length])
  sense_indices = tf.placeholder(tf.int32, [(context_window*2+batch_size) * sense_dim])

  # [(context_window*2+batch_size), max_context_length, embedding_dim]
  embedded_context = tf.nn.embedding_lookup(w_in, (context_indices))
  embedded_context = tf.transpose(embedded_context, perm = [0,2,1])
  embedded_context = tf.reduce_sum(embedded_context,2,keep_dims=True)

  # [(context_window*2+batch_size) * sense_dim, embedding_dim]
  embedded_word_output = tf.nn.embedding_lookup(w_out, (sense_indices))
  embedded_word_output = tf.reshape(embedded_word_output, [(context_window*2+batch_size), sense_dim, embedding_dim])

  # shape = [(context_window*2+batch_size), sense_dim, 1]
  sense_score = tf.batch_matmul(embedded_word_output, embedded_context)
  # [(context_window*2+batch_size), sense_dim]
  sense_score = tf.squeeze(sense_score)
  
  # [context_window*2+batch_size]
  sense_greedy = tf.to_int32(tf.argmax(sense_score, 1))

  target_sense_sampled_indices = tf.placeholder(tf.int32, [batch_size])
  collocation_sense_sampled_indices = tf.placeholder(tf.int32, [batch_size])

  # [batch_size]
  reward_prob = tf.placeholder(tf.float32, [batch_size], name='reward_logit')
  reward_log_L = tf.log(reward_prob)
  #reward = tf.sigmoid(reward_logit)

  if formulation == 'value_function_factorization':
    forward_state_value = tf.Variable(tf.zeros([vocab_size, embedding_dim]),\
                trainable=True, name="forward_state_value")
    backward_state_value = tf.Variable(tf.zeros([vocab_size, embedding_dim]),\
                trainable=True, name="backward_state_value")
    # [context_window*2+batch_size]
    word_indices = tf.gather(tf.transpose(context_indices, perm = [1, 0]), context_window)
    # [context_window*2+batch_size, embedding_size]
    f_state_embedding = tf.nn.embedding_lookup(forward_state_value, word_indices)
    b_state_embedding = tf.nn.embedding_lookup(backward_state_value, word_indices)
    embedded_context = tf.squeeze(embedded_context)
    # [context_window*2+batch_size]
    f_state_value_score = tf.reduce_sum(tf.mul(embedded_context, f_state_embedding), 1)
    b_state_value_score = tf.reduce_sum(tf.mul(embedded_context, b_state_embedding), 1)
    # [sense_dim, (context_window*2+batch_size)]
    sense_score = tf.transpose(sense_score, perm = [1,0])
    # [(context_window*2+batch_size), sense_dim]
    f_sense_value_score = tf.transpose(tf.add(sense_score, f_state_value_score), perm = [1,0])
    b_sense_value_score = tf.transpose(tf.add(sense_score, b_state_value_score), perm = [1,0])
    sense_prob = tf.nn.softmax(f_sense_value_score)

    f_sense_value_score = tf.reshape(f_sense_value_score, [(context_window*2+batch_size) * sense_dim])
    b_sense_value_score = tf.reshape(b_sense_value_score, [(context_window*2+batch_size) * sense_dim])
    sense_selected_logit_input = tf.gather(f_sense_value_score, target_sense_sampled_indices)
    sense_selected_logit_output = tf.gather(b_sense_value_score, collocation_sense_sampled_indices)
    print 'value_function_factorization'
  else:
    # [(context_window*2+batch_size), sense_dim]
    sense_prob = tf.nn.softmax(sense_score)
    # [(context_window*2+batch_size)* sense_dim]
    sense_score = tf.reshape(sense_score, [(context_window*2+batch_size)* sense_dim])
    # [batch_size]
    sense_selected_logit_input = tf.gather(sense_score, target_sense_sampled_indices)
    sense_selected_logit_output = tf.gather(sense_score, collocation_sense_sampled_indices)
    print 'no value_function_factorization'

  sense_prob_flat = tf.reshape(sense_prob, [(context_window*2+batch_size) * sense_dim])
  # [batch_size]
  sense_selected_logP_input = tf.log(tf.gather(sense_prob_flat, target_sense_sampled_indices))
  sense_selected_logP_output = tf.log(tf.gather(sense_prob_flat, collocation_sense_sampled_indices))

  # [batch_size, sense_dim]
  #sense_input_logit, sense_output_logit = tf.split(0,2,sense_greedy_logit_wordNetwork)
  if learning_framework == 'value_based':
    if formulation == 'original_formulation' or formulation == 'value_function_factorization':
      print 'value based, two-sided optimization'
      cost = tf.div(tf.add(tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(sense_selected_logit_input, reward_prob)), tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(sense_selected_logit_output, reward_prob))), 2.)
    elif formulation == 'one-sided_optimization':
      cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(sense_selected_logit_input, reward_prob))
      print 'value based, one-sided optimization'
  elif learning_method == 'policy':
    if formulation == 'original_formulation':
      cost = tf.neg(tf.div(tf.reduce_mean(tf.add(tf.mul(reward_log_L, sense_selected_logP_input), tf.mul(reward_log_L,sense_selected_logP_output))), 2))
      print 'policy gradient, two-sided optimization'
    else:
      cost = tf.neg(tf.reduce_mean(tf.mul(reward_log_L, sense_selected_logP_input)))
      print 'policy gradient, one-sided optimization'
  else:
    if formulation == 'original_formulation':
      print 'smooth policy gradient, two-sided optimization'
      cost = tf.neg(tf.div(tf.reduce_mean(tf.add(tf.mul(reward_prob, sense_selected_logP_input), tf.mul(reward_prob,sense_selected_logP_output))), 2))
    else:
      print 'smooth policy gradient, one-sided optimization'
      cost = tf.neg(tf.reduce_mean(tf.mul(reward_prob, sense_selected_logP_input)))

  print '==================================='
  print 'learning method:', learning_method
  print 'learning framework:', learning_framework
  print 'formulation:', formulation
  print '==================================='

  optimizer = tf.train.GradientDescentOptimizer(learning_rate)
  update = optimizer.minimize(cost)

  init = tf.initialize_all_variables()

saver = tf.train.Saver(tf.all_variables(), max_to_keep=None)
best_saver = tf.train.Saver(tf.all_variables(), max_to_keep=1)
best_avgC_saver = tf.train.Saver(tf.all_variables(), max_to_keep=1)



def getEmbedding(sess,testData, testLocation):
  senseVec = [[] for i in xrange(len(testData))]
  senseScore = [[] for i in xrange(len(testData))]

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
    sense_probability = sess.run(sense_prob, feed_dict={context_indices: c_indices, sense_indices: s_indices})
   
    #print sense_probability, 'leaving sess'

    for k in xrange(sense_dim):
      selected_s_input_indices = []
      for i in xrange(batch_size):
        if currentIndex-batch_size+i < len(testData):
          wordIndex = testData[currentIndex-batch_size+i][testLocation[currentIndex-batch_size+i]]
          selected_s_input_indices.append(wordIndex*sense_dim+ k)
        else:
          selected_s_input_indices.append(0)
      embedded_sense = sess.run(embedded_sense_input, feed_dict={selected_sense_input_indices: selected_s_input_indices})
      
      for i in xrange(batch_size):
        index = currentIndex-batch_size+i
        #print index, i, batch_size, len(testData), len(senseVec), len(embedded_sense)
        if index < len(testData):
          senseVec[index].append(embedded_sense[i])

    selected_w_input_indices = []
    for i in xrange(batch_size):
      if currentIndex-batch_size+i < len(testData):
        wordIndex = testData[currentIndex-batch_size+i][testLocation[currentIndex-batch_size+i]]
        selected_w_input_indices.append(wordIndex)
      else:
        selected_w_input_indices.append(0)
    
    for i in xrange(batch_size):
      index = currentIndex-batch_size+i
      if index < len(testData):
        senseScore[index] = sense_probability[i]

  return np.asarray(senseVec), np.asarray(senseScore)

def get_train_batch(data, leftBoundry, rightBoundry):
  c_indices = [None] * (context_window*2+batch_size)
  s_indices = [None] * ((context_window*2+batch_size)*sense_dim)

  indices = range(context_window, context_window*3+batch_size)

  dynamic_window = np.random.randint(1, context_window+1, len(indices))
  for i in xrange(len(indices)):
    leftBoundry[indices[i]] = min(leftBoundry[indices[i]], dynamic_window[i])
    rightBoundry[indices[i]] = min(rightBoundry[indices[i]], dynamic_window[i])
    c_indices[i] = [PAD_ID] * (context_window - leftBoundry[indices[i]]) + data[indices[i]-leftBoundry[indices[i]]:indices[i]+1+rightBoundry[indices[i]]] + [PAD_ID] * (context_window - rightBoundry[indices[i]])
    
  for i in xrange(len(indices)):
    offset = data[indices[i]]*sense_dim
    for k in xrange(sense_dim):
      s_indices[i*sense_dim+k] = offset + k

  input_feed = {context_indices: c_indices, sense_indices: s_indices}
 
  return input_feed, dynamic_window

def get_train_sample_batch(sense_selected, input_feed, dynamic_window):
  # [context_window*2+batch_size]
  c_indices = input_feed[context_indices]
  collocation_selected = []
  target_selected = []

  selected_s_input_indices = []
  selected_s_output_indices = []
  foundN = 0 

  for i in xrange(context_window, batch_size+context_window):
    index = c_indices[i][context_window]*sense_dim+sense_selected[i]
    target_selected.append(i*sense_dim + sense_selected[i])
    found = False

    left = context_window*2
    right = 0

    for t in xrange(context_window*2+1):
      if t == context_window or c_indices[i][t] == PAD_ID:
        continue
      else:
        if t < left:
          left = t
        if t > right:
          right = t

    selected_s_input_indices.append(index)
    if left > right:
      print left, right
      print c_indices[i]
    assert(left <= right)
    if left < right:
      t = randint(left+1,right)
    else:
      t = left
    if t == context_window:
      t = left
    t_index = i + (t - context_window)
    collocation_selected.append(t_index*sense_dim + sense_selected[t_index])

    t_index = c_indices[t_index][context_window]*sense_dim+sense_selected[t_index]
    selected_s_output_indices.append(t_index)

  input_feed[selected_sense_input_indices] = selected_s_input_indices
  input_feed[selected_sense_output_indices] = selected_s_output_indices

  return input_feed, collocation_selected, target_selected

def evaluate(sess):
  with Timer("get embedding for 1st half of evaluation data"):
    [senseVec, senseScore] = getEmbedding(sess,testData, testLocation)

  indices1 = range(0,len(senseVec),2)
  indices2 = range(1,len(senseVec),2)

  avgSimC = calAvgSimC(test_score, senseVec[indices1], senseScore[indices2], senseVec[indices2], senseScore[indices2])
  maxSimC = calMaxSimC(test_score, senseVec[indices1], senseScore[indices2], senseVec[indices2], senseScore[indices2])
  
  scores = [maxSimC, avgSimC]
  score_name = ['MaxSimC', 'AvgSimC']
  
  print 'AvgSimC =','{:.5f}'.format(avgSimC), 'MaxSimC =','{:.5f}'.format(maxSimC)
  return [scores, score_name]

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.memory)
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
  sess.run(init)
  sess.run(init_word2vec)

  # Training cycle
  best_eval_sim = 0
  best_eval_loss = 1e10
  start_time = time.time()

  with open(log_path,'w') as fp:
    fp.write('epoch MaxSimC AvgSimC total_time\n')
  
  data = [PAD_ID for i in xrange(context_window*2)]
  leftBoundry = [0 for i in xrange(context_window*2)]
  rightBoundry = [0 for i in xrange(context_window*2)]
  nData = context_window
  total_batch = trainDataSize/batch_size
  rnd_index = 0
  ite = 0

  bestLocal = args.ckpt_threshold
  bestAvgC = args.ckpt_threshold

  for epoch in range(training_epochs):
    avg_loss = 0.
    
    train_start_time = time.time()
    scores, score_name = evaluate(sess)
    
    evaluation_time = time.time()-start_time
    with open(log_path,'a') as fp:
      fp.write('completed epoch ')
      epoch_sting = '%05d' % (epoch)
      total_time = '%05d' % (evaluation_time/60)
      outputString = []
      outputString.append(epoch_sting)
      for s in scores:
        outputString.append('{:.4f}'.format(s))
      outputString.append(total_time)

      fp.write(' '.join(outputString)+'\n')
      #fp.write('complete epoch\n')
      if epoch != 0:
        save_path = save_dir + "lr_"+str(lr_rate)+"_window_"+str(context_window)+"_batch_"+str(batch_size)+"_sample_"+str(samp_size)+"_sense_"+str(sense_dim)+'-'+str(epoch)
        save_path = saver.save(sess, save_path)
        print "\nModel saved in file: %s\n" % save_path

    getTrainBatchTime = 0
    senseProbTime = 0
    getWord2vecBatch = 0
    word2vecTime = 0
    pgTime = 0
    
    with open(dataset_prefix+'/train.sensplit.txt') as fp:
      for e,line in enumerate(fp):
        if e % print_interval == 0:
          print e,'/ 42810886 line, current iteration:', ite
        try:
          line = map(int,line.strip().split(' '))
        except:
          continue
        # subsampling
        offset = len(data)
        counter = 0
        for i in xrange(len(line)):
          rnd_index = (rnd_index+1) % len(rnd)
          if id2freq[line[i]] < rnd[rnd_index]:
            continue
          #nData+=1
          data.append(line[i])
          leftBoundry.append(min(context_window,counter))
          counter+=1
        lineLength = len(data)-offset
        for i in xrange(lineLength):
          rightBoundry.append(min(context_window,lineLength-1-i))
        if lineLength == 1:
          data = data[:-1]
          leftBoundry = leftBoundry[:-1]
          rightBoundry = rightBoundry[:-1]

        assert(len(data) == len(leftBoundry))
        assert(len(data) == len(rightBoundry))

        if len(data) < batch_size+context_window*4:
          continue
        else:
          ite+=1

        if ite*batch_size % (500*2048) == 0:
          scores, score_name = evaluate(sess)
          if bestLocal < scores[0]:
            save_path = save_dir + "lr_"+str(lr_rate)+"_window_"+str(context_window)+"_batch_"+str(batch_size)+"_sample_"+str(samp_size)+"_sense_"+str(sense_dim)+'-bestMaxC-'+str(ite)
            save_path = best_saver.save(sess, save_path)
            bestLocal = scores[0]
            print "\n========best MaxSimC model saved in file: %s========\n" % save_path
          if bestAvgC < scores[1]:
            save_path = save_dir + "lr_"+str(lr_rate)+"_window_"+str(context_window)+"_batch_"+str(batch_size)+"_sample_"+str(samp_size)+"_sense_"+str(sense_dim)+'-bestAvgC-'+str(ite)
            save_path = best_avgC_saver.save(sess, save_path)
            bestAvgC = scores[1]
            print "\n========best AvgSimC model saved in file: %s========\n" % save_path
          with open(log_path,'a') as fp:
            evaluation_time = time.time()-start_time
            epoch_sting = '{:.5f}'.format(ite/float(total_batch))
            evaluation_time = time.time()-start_time
            total_time = '%05d' % (evaluation_time/60)
            outputString = []
            outputString.append(epoch_sting)
            for s in scores:
              outputString.append('{:.4f}'.format(s))
            outputString.append(total_time)

            fp.write(' '.join(outputString)+'\n')

          print '{:.5f}'.format(ite/float(total_batch)), 'epoch', ite, 'iteration'
          
        lastTime = time.time()
        input_feed, dynamic_window = get_train_batch(data[:batch_size+context_window*4], leftBoundry[:batch_size+context_window*4], rightBoundry[:batch_size+context_window*4])
        data = data[batch_size:]
        leftBoundry = leftBoundry[batch_size:]
        rightBoundry = rightBoundry[batch_size:]

        if learning_framework == 'policy_gradient' or learning_method == 'Q-Boltzmann':
          sense_probability = sess.run(sense_prob, feed_dict=input_feed)
          sense_selected = np.zeros([(context_window*2+batch_size)], dtype=np.int32)
          candidate = range(sense_dim)
          for i in xrange(len(sense_selected)):
            sense_selected[i] = choice(candidate, p=sense_probability[i])
        elif learning_method == 'Q-epsilon-greedy':
          sense_greedy_choice = sess.run(sense_greedy, feed_dict=input_feed)
          sense_selection_mask = np.random.rand(context_window*2+batch_size) < 0.05
          sense_selected = sense_selection_mask * np.random.randint(0,sense_dim, context_window*2+batch_size) \
            + (1-sense_selection_mask) * sense_greedy_choice
        else:
          sense_selected = sess.run(sense_greedy, feed_dict=input_feed)

        #word2vec_feed = get_train_sample_batch(sense_selected, input_feed, dynamic_window)
        word2vec_feed, collocation_selected, target_selected = get_train_sample_batch(sense_selected, input_feed, dynamic_window)
        
        _, word_processed = sess.run([train, total_words_processed], feed_dict=word2vec_feed)
        reward = sess.run(reward_sense_prob, feed_dict=word2vec_feed)

        input_feed[reward_prob] = reward
        input_feed[collocation_sense_sampled_indices] = collocation_selected
        input_feed[target_sense_sampled_indices] = target_selected
        
        _, c  = sess.run([update, cost], feed_dict=input_feed)

    evaluation_time = time.time()-start_time

    print "\n===Epoch:", '%04d' % (epoch+1), \
        "training time=",'%04d min===\n' % (evaluation_time/60)
