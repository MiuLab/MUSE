from datetime import datetime
import gzip, cPickle, pdb, os

_PAD = "<pad>"
PAD_ID = 0

_UNK = "<unk>"
UNK_ID = 1

_END = "end.of.document"
END_ID = 2

_BR = "<br>"
BR_ID = 3

def save_pklgz( filename, obj):
	pkl = cPickle.dumps( obj, protocol = cPickle.HIGHEST_PROTOCOL);
	with gzip.open( filename, "wb") as fp:
		fp.write( pkl);

def load_pklgz( filename):
	print("... loading", filename);
	with gzip.open( filename, 'rb') as fp:
		obj = cPickle.load(fp);
	return obj;

class Timer(object):
	def __init__(self, name=None, verbose=2):
		self.name = name
		self.verbose = verbose;

	def __enter__(self):
		if self.name and self.verbose >= 1:
			print("...", self.name)
		self.start = datetime.now()
		return self

	def __exit__(self, type, value, traceback):
		if self.verbose >= 2:
			if self.name:
				print('...', self.name, "done in", datetime.now() - self.start)
			else:
				print("Elapsed:", datetime.now() - self.start)
