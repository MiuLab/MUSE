from utils import save_pklgz, load_pklgz, PAD_ID, UNK_ID, _UNK, _PAD, _END, END_ID, _BR, BR_ID, Timer
import sys
import time
from collections import defaultdict
from nltk.tokenize import sent_tokenize
word2id = dict()

def extractLocation(context,word):
	line = context.strip().split(' ')
	location = -1
	for j in xrange(len(line)):
		if line[j] == '<b>':
			location = j
			line = line[:j]+line[j+1]+line[j+3:]
			break
	if line[location]!=word:
		print line[location], word
	return location, line

if len(sys.argv) != 5:
	print 'usage: python makeTrain.py train.in train.out'
trainInFile = sys.argv[1]
trainOutFile = sys.argv[2]

wordCount = defaultdict(int)
with open('wordID_MSSG.txt') as fp:
	for line in fp:
		line = line.strip().split(' ')
		word = line[0]
		ID = line[1]
		word2id[word] = ID

start_time = time.time()
dataN = 0
uDataN = 0
with Timer('formating training data'):
	with open(trainInFile) as infp, open(trainOutFile,'w') as outfp, open('temp.txt','w') as tempoutfp:
		for e,line in enumerate(infp):
			print e
			if (e+1) % 30 == 0:
				print e+1, time.time()-start_time, dataN+uDataN, dataN/float(dataN+uDataN)
				#exit(0)
				#print wordCount['<unk>']
			line = line.strip().lower()
			sentences = sent_tokenize(line)
			for e,sentence in enumerate(sentences):
				
				sentence = sentence.strip().split(' ')
				if sentence[0] == '<br>':
					sentence = sentence[1:]
				#tempoutfp.write(' '.join(sentence)+'\n')
				if len(sentence) <= 10:
					#print sentence,'\n'
					#if e-1>=0:
					#	print sentences[e-1]
					#if e+1<len(sentences):
					#	print sentences[e+1]
					#print '====\n'
					continue
				data = []
				tempdata = []
				for word in sentence:
					if word in word2id and word!=_BR and word!=_END and word!=_UNK:
						data.append(word2id[word])
						tempdata.append(word)
						wordCount[word]+=1
						dataN+=1
					else:
						uDataN+=1
						continue

				data = map(str,data)
				outfp.write(' '.join(data)+'\n')
				tempoutfp.write(' '.join(tempdata)+'\n')

print '# of data:',dataN
print '# of unk data:',uDataN
with open('wordID_MSSG.txt') as fp, open('wordID_MSSG_unigram_sensplit.txt','w') as outfp:
	for line in fp:
		line = line.strip().split(' ')
		word = line[0]
		ID = line[1]
		outfp.write(' '.join(line)+' '+str(wordCount[word])+'\n')
