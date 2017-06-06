import sys
from scipy.stats import spearmanr, pearsonr
from scipy.spatial.distance import cosine
import numpy as np
from utils import save_pklgz, load_pklgz, PAD_ID, UNK_ID, _UNK, _PAD, _END, END_ID, _BR, BR_ID, Timer
from collections import defaultdict
filename = 'vectors.MSSG.300D.6K'
word2ID = dict()
word2ID[_PAD] = PAD_ID
word2ID[_UNK] = UNK_ID
word2ID[_END] = END_ID
word2ID[_BR] = BR_ID
globalVec = dict()
contextVec = dict()
senseVec = dict()
wordSet = set()
words = []
senseCount = defaultdict(int)
with open(filename) as fp, open('wordID_MSSG.txt','w') as outfp:
	line = fp.readline()
	line = line.strip().split(' ')
	nWord = int(line[0])
	D = int(line[1])
	outfp.write(_PAD+' '+str(PAD_ID)+' 1\n')
	outfp.write(_UNK+' '+str(UNK_ID)+' 1\n')
	outfp.write(_END+' '+str(END_ID)+' 1\n')
	outfp.write(_BR+' '+str(BR_ID)+' 1\n')
	while True:
		line = fp.readline()
		if not line:
			break
		line = line.strip().split(' ')
		word = line[0]
		wordSet.add(word)
		words.append(word)
		if len(wordSet) % 1000 == 0:
			print len(wordSet), 'words'
			print word
		#print word
		nSense = int(line[1])
		senseCount[nSense]+=1
		if nSense!=3 and nSense!=1:
			print word, nSense
		word2ID[word] = str(len(word2ID))
		outfp.write(word+' '+word2ID[word]+' '+line[1]+'\n')
		globalVec[word] = map(float, fp.readline().strip().split(' '))
		contextVec[word] = []
		senseVec[word] = []
		for i in xrange(nSense):
			senseVec[word].append(map(float, fp.readline().strip().split(' ')))
			contextVec[word].append(map(float, fp.readline().strip().split(' ')))

