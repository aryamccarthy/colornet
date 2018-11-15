import itertools
import numpy
import gensim
from pathlib import Path
import pickle
import os
import sys

# input paths
MODEL_PATH = "/Users/ryanculkin/Downloads/GoogleNews-vectors-negative300.bin"
OUT_PATH = "/Users/ryanculkin/Desktop/dl/proj/subset.p"
CLR_DIR = Path("../../data/raw")
QUANT_PATH = '%s/quantifiers.txt' % (CLR_DIR)
CMP_PATH = '%s/comparatives.txt'  % (CLR_DIR)

# constants
LESS = 'less'
MORE = 'more'
PAD = '<PAD>'

def extract_words(path):
	with open(path, 'r') as f:
		lines = [i.strip() for i in f.readlines()]
		key_value_pairs = [i.split(':') for i in lines]
		words = list(itertools.chain.from_iterable(key_value_pairs))
		return set(words)


if __name__ == '__main__':
	if os.path.exists(OUT_PATH):
		print ('file at %s exists... testing deserialization and then exiting' % (OUT_PATH))
		with open(OUT_PATH, 'r') as in_f:
			obj = pickle.load(in_f)
		print ('success; exiting.')
		sys.exit(0)

	# 1: determine the vocab
	quantifiers = extract_words(QUANT_PATH)
	comparatives = extract_words(CMP_PATH)
	vocab = set(quantifiers).union(set(comparatives))

	vocab.add(LESS)
	vocab.add(MORE)
	print ('vocab size:%d' % (len(vocab)))

	# 2: get the word embeddings for the words in the vocabulary
	model = gensim.models.KeyedVectors.load_word2vec_format(MODEL_PATH, binary=True)
	embedding_dict = {i : model.wv[i] for i in vocab if i in model.wv}
	oovs = [i for i in vocab if i not in model.wv]

	print ('words not found in word2vec: %s' % (str(oovs)))

	# manually add a special 0-vector for "<PAD>"
	dimensionality = model.wv[MORE].shape
	embedding_dict[PAD] = numpy.zeros(dimensionality)

	# save the dictionary
	with open(OUT_PATH, 'w') as out_f:
		pickle.dump(embedding_dict, out_f)

