# -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 11:28:51 2018

@author: ankish
"""

import string
import re
from pickle import dump
from unicodedata import normalize

from pickle import load
from collections import Counter


# In[]:

# load doc into memory
def load_doc(filename):
	# open the file as read only
	file = open(filename, mode='rt', encoding='utf-8')
	# read all text
	text = file.read()
	# close the file
	file.close()
	return text

# split a loaded document into sentences
def to_sentences(doc):
	return doc.strip().split('\n')


# shortest and longest sentence lengths
def sentence_lengths(sentences):
	lengths = [len(s.split()) for s in sentences]
	return min(lengths), max(lengths)

'''

    Tokenizing text by white space.
    Normalizing case to lowercase.
    Removing punctuation from each word.
    Removing non-printable characters.
    Converting French characters to Latin characters.
    Removing words that contain non-alphabetic characters.

'''


# clean a list of lines
def clean_lines(lines):
	cleaned = list()
	# prepare regex for char filtering
	re_print = re.compile('[^%s]' % re.escape(string.printable))
	# prepare translation table for removing punctuation
  # str.maketrans('abc','123') means mapping 'a' -> '1', 'b' -> '2'
  
	table = str.maketrans('', '', string.punctuation)
	for line in lines:
		# normalize unicode characters
		line = normalize('NFD', line).encode('ascii', 'ignore')
		line = line.decode('UTF-8')
		# tokenize on white space
		line = line.split()
		# convert to lower case
		line = [word.lower() for word in line]
		# remove punctuation from each token
		line = [word.translate(table) for word in line]
		# remove non-printable chars form each token
		line = [re_print.sub('', w) for w in line]
		# remove tokens with numbers in them
		line = [word for word in line if word.isalpha()]
		# store as string
		cleaned.append(' '.join(line))
	return cleaned

# save a list of clean sentences to file
def save_clean_sentences(sentences, filename):
	dump(sentences, open(filename, 'wb'))
	print('Saved: %s' % filename)

# In[]:
# load English data
filename = 'D:/Tensorflowspyder/NLP/fr-en.tgz/fr-en/europarl-v7.fr-en.en'
doc = load_doc(filename)
sentences = to_sentences(doc)
sentences = clean_lines(sentences)
save_clean_sentences(sentences, 'D:/Tensorflowspyder/NLP/data/english.pkl')
# spot check
for i in range(10):
	print(sentences[i])
# D:\Tensorflowspyder\NLP\data
    
    
# In[]:
    
# load French data
filename = 'D:/Tensorflowspyder/NLP/fr-en.tgz/fr-en/europarl-v7.fr-en.fr'
doc = load_doc(filename)
sentences = to_sentences(doc)
sentences = clean_lines(sentences)
save_clean_sentences(sentences, 'D:/Tensorflowspyder/NLP/data/french.pkl')
# spot check
for i in range(10):
	print(sentences[i])

# In[]:


# load a clean dataset
def load_clean_sentences(filename):
	return load(open(filename, 'rb'))

# save a list of clean sentences to file
def save_clean_sentences(sentences, filename):
	dump(sentences, open(filename, 'wb'))
	print('Saved: %s' % filename)

# create a frequency table for all words
def to_vocab(lines):
	vocab = Counter()
	for line in lines:
		tokens = line.split()
		vocab.update(tokens)
	return vocab

# remove all words with a frequency below a threshold
def trim_vocab(vocab, min_occurance):
	tokens = [k for k,c in vocab.items() if c >= min_occurance]
	return set(tokens)

# mark all OOV with "unk" for all lines
def update_dataset(lines, vocab):
	new_lines = list()
	for line in lines:
		new_tokens = list()
		for token in line.split():
			if token in vocab:
				new_tokens.append(token)
			else:
				new_tokens.append('unk')
		new_line = ' '.join(new_tokens)
		new_lines.append(new_line)
	return new_lines

# In[]:
# load English dataset
filename = 'D:/Tensorflowspyder/NLP/data/english.pkl'
lines = load_clean_sentences(filename)
# calculate vocabulary
vocab = to_vocab(lines)
print('English Vocabulary: %d' % len(vocab))
# reduce vocabulary
vocab = trim_vocab(vocab, 5)
print('New English Vocabulary: %d' % len(vocab))
# mark out of vocabulary words
lines = update_dataset(lines, vocab)
# save updated dataset
filename = 'english_vocab.pkl'
save_clean_sentences(lines, filename)
# spot check
for i in range(10):
	print(lines[i])

# In[]:
# load French dataset
filename = 'D:/Tensorflowspyder/NLP/data/french.pkl'
lines = load_clean_sentences(filename)
# calculate vocabulary
vocab = to_vocab(lines)
print('French Vocabulary: %d' % len(vocab))
# reduce vocabulary
vocab = trim_vocab(vocab, 5)
print('New French Vocabulary: %d' % len(vocab))
# mark out of vocabulary words
lines = update_dataset(lines, vocab)
# save updated dataset
filename = 'french_vocab.pkl'
save_clean_sentences(lines, filename)
# spot check
for i in range(10):
	print(lines[i])