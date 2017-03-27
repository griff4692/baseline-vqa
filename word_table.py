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
			os.path.isfile('./data/word2Idx')
			self.load_dictionary()
		except IOError:
			self.generate_dictionary(processed_data_file)

	def generate_dictionary(self, processed_data_file, serialize=True):
		# word2idx dictionary
		self.word2Idx = {}

		# idx2word array
		self.idx2Word = []

		with open(processed_data_file, 'rb') as csvdata:
			data = csv.reader(csvdata, delimiter='~')

			for (_, question, answer) in data:
				words = question + ' ' + answer
				for word in re.split(r'[^\w]+', words):
					self.add_word(word)


		csvdata.close()

		self.vocab_size = len(self.idx2Word)

		if serialize:
			fd = open('./data/idx2Word', 'w')
			pickle.dump(self.idx2Word, fd)
			fd.close()

			fd = open('./data/word2Idx', 'w')
			pickle.dump(self.word2Idx, fd)
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

	def add_word(self, word):
		lc = word.lower()
		if lc not in self.word2Idx:
			idx = len(self.idx2Word)
			self.idx2Word.append(lc)
			self.word2Idx[lc] = idx

	def load_dictionary(self):
		fd = open('./data/idx2Word', 'r')
		self.idx2Word = pickle.load(fd)
		self.vocab_size = len(self.idx2Word)
		fd.close()
		fd = open('./data/word2Idx', 'r')
		self.word2Idx = pickle.load(fd)
		fd.close()

	def getLabelIdx(self, label):
		label = label.lower()

		if label in self.labels2Idx:
			return self.labels2Idx[label]
		else:
			return -1

	def vocabSize(self):
		return len(self.idx2Word)

	def getIdx(self, word):
		word = word.lower()
		if word not in self.word2Idx:
			return self.vocabSize()

		return self.word2Idx[word]

	def getWord(self, idx):
		return self.idx2Word[idx]

	def getBOW(self, str):
		bow = np.zeros(self.vocab_size)

		words = re.split(r'[^\w]+', str)

		for word in words:
			bow[self.getIdx(word)] += 1

		return bow

	def encodeQ(self,str):
		words = re.split(r'[^\w]+', str)
		res = []
		for word in words:
			res.append(self.getIdx(word))
		return res



