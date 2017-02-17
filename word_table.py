import os
from preprocess import process_json, get_default_data_path, get_default_processed_data_path
import csv
import re
import pickle
import numpy as np
from collections import defaultdict

class WordTable:

	def __init__(self):
		default_data_path = get_default_data_path()
		# relies on having csv parsed file
		try:
			os.path.isfile(default_data_path)
			processed_data_file = get_default_processed_data_path()
		except IOError:
			processed_data_file = process_json()

		try:
			os.path.isfile('./data/idx2word')
			os.path.isfile('./data/word2idx')
			self.load_dictionary()
		except IOError:
			self.generate_dictionarys(processed_data_file)

	def generate_dictionarys(self, processed_data_file, serialize=True):
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


		csvdata.close()

		self.vocab_size = len(self.idx2word)

		if serialize:
			fd = open('./data/idx2word', 'w')
			pickle.dump(self.idx2word, fd)
			fd.close()

			fd = open('./data/word2idx', 'w')
			pickle.dump(self.word2idx, fd)
			fd.close()

	def top_answers(self, processed_data_file, max_classes=1000):
		answers = defaultdict(int)

		with open(processed_data_file, 'rb') as csvdata:
			data = csv.reader(csvdata, delimiter='~')
			for (_, _, answer) in data:				
				answers[answer.lower()] += 1


		csvdata.close()

		sorted_answers = sorted(answers, key=answers.get, reverse=True)
		self.labels = sorted_answers[0:max_classes]

		self.labels2Idx = {}

		for i in range(len(self.labels)):
			self.labels2Idx[self.labels[i]] = i

		return self.labels2Idx


	def load_dictionary(self):
		fd = open('./data/idx2word', 'r')
		self.idx2Word = pickle.load(fd)
		self.vocab_size = len(self.idx2Word)
		fd.close()
		fd = open('./data/word2idx', 'r')
		self.word2Idx = pickle.load(fd)
		fd.close()

	def vocabSize(self):
		return len(self.idx2Word)

	def getIdx(self, word):
		word = word.lower()
		if word in self.word2Idx:
			return self.word2Idx[word]
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



