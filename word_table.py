import os
from preprocess import process_json, get_default_data_path, get_default_processed_data_path
import csv
import re
import pickle
import numpy as np


class WordTable:

	def __init__(self):
		# relies on having csv parsed file
		try:
			default_data_path = get_default_data_path()
			os.path.isfile(default_data_path)
			processed_data_file = get_default_processed_data_path()
		except IOError:
			processed_data_file = process_json()

		self.generate_dictionary(processed_data_file, False)

	def generate_dictionary(self, processed_data_file, serialize=False):
		# word2idx dictionary
		self.word2idx = {}

		# idx2word array
		self.idx2word = []

		with open(processed_data_file, 'rb') as csvdata:
			data = csv.reader(csvdata, delimiter='~')

			for (_, question, answer) in data:
				words = question + ' ' + answer
				for word in re.split(r'[^\w]+', words):
					lc = word.lower()
					if lc not in self.word2idx:
						idx = len(self.idx2word)
						self.idx2word.append(lc)
						self.word2idx[lc] = idx


		self.vocab_size = len(self.idx2word)


		if serialize:
			fd = open('./data/idx2word', 'w')
			pickle.dump(self.idx2word, fd)
			fd.close()

			fd = open('./data/word2idx', 'w')
			pickle.dump(self.word2idx, fd)
			fd.close()


	def load_dictionary(self):
		fd = open('./data/idx2word', 'r')
		idx2Word = pickle.load(fd)
		fd.close()
		fd = open('./data/word2idx', 'r')
		word2Idx = pickle.load(fd)
		fd.close()

	def getIdx(self, word):
		word = word.lower()
		if word in self.word2idx:
			return self.word2idx[word]
		else:
			return -1

	def getWord(self, idx):
		return self.idx2word[idx]

	def getBOW(self, str):
		bow = np.zeros(self.vocab_size)

		words = re.split(r'[^\w]+', str)

		for word in words:
			idx = self.getIdx(word)
			if idx > -1:
				bow[idx] += 1

		return bow



