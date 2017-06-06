#!/usr/bin/env python
# -*- coding: utf-8 -*-
from nltk.tokenize import StanfordTokenizer
import time

def is_ascii(s):
	return all(ord(c) < 128 for c in s)

last_time = time.time()
line_buffer = ''
with open('WestburyLab.Wikipedia.Corpus.txt') as infp, open('TokenizedCorpus.txt','w') as outfp:
	for e,line in enumerate(infp):
		if (e+1) % 10000 == 0:
			line_buffer = StanfordTokenizer().tokenize(line_buffer)
			try:
				outfp.write(' '.join(line_buffer)+'\n')
			except:
				for i in xrange(len(line_buffer)):
					if not is_ascii(line_buffer[i]):
						line_buffer[i] = '<UNK>'
				outfp.write(' '.join(line_buffer)+'\n')
			line_buffer = ''
			print e+1, '/ 30749930', float(e+1)/30749930,time.time()-last_time

		if line.strip() == '':
			continue
		line_buffer += (line+' <br> ')

	line_buffer = StanfordTokenizer().tokenize(line_buffer)
	try:
		outfp.write(' '.join(line_buffer)+'\n')
	except:
		for i in xrange(len(line_buffer)):
			if not is_ascii(line_buffer[i]):
				line_buffer[i] = '<UNK>'
		outfp.write(' '.join(line_buffer)+'\n')
