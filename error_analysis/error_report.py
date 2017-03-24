import sys
import csv
import os

class Error:
	def __init__(self, value, error_type):
		self.num_correct = 0
		self.num_tested = 0

		self.value = value

		self.type = error_type
		
		if self.type == 'question':
			self.guessed_labels = {}
			self.true_labels = {}

		if self.type == 'true_label':
			self.guessed_labels = {}
			self.question_unigrams = {}
			self.question_bigrams = {}

		if self.type == 'guessed_label':
			self.true_labels = {}
			self.question_unigrams = {}
			self.question_bigrams = {}


	def incr_true_labels(self, true_label):
		if true_label not in self.true_labels:
			self.true_labels[true_label] = 1
		else:
			self.true_labels[true_label] += 1


	def incr_correct(self, is_correct):
		self.num_tested += 1
		if is_correct:
			self.num_correct += 1

	def incr_guessed_labels(self, guessed_label):
		if guessed_label not in self.guessed_labels:
			self.guessed_labels[guessed_label] = 1
		else:
			self.guessed_labels[guessed_label] += 1

	def incr_question_unigrams(self, unigram):
		if unigram not in self.question_unigrams:
			self.question_unigrams[unigram] = 1
		else:
			self.question_unigrams[unigram] += 1

	def incr_question_bigrams(self, bigram):
		if bigram not in self.question_bigrams:
			self.question_bigrams[bigram] = 1
		else:
			self.question_bigrams[bigram] += 1			


def generate_report(model_name):
	DATA_DIR = './results/' + model_name +'/'

	unigrams = {}
	bigrams = {}

	true_labels = {}
	guessed_labels = {}

	num_tested = 0
	num_correct = 0

	with open(DATA_DIR + 'errors.csv', 'rb') as csvfile:
		rows = csv.reader(csvfile, delimiter='~')

		for row in rows:
			q = row[0]
			q_words = q.split()

			unigram = q_words[0]
			bigram = unigram + ' ' + q_words[1]

			guessed_label = row[1]
			true_label = row[2]

			is_correct = compare(guessed_label, true_label)

			num_tested += 1
			if is_correct:
				num_correct += 1

			if unigram not in unigrams:
				unigrams[unigram] = Error(unigram, 'question')

			if bigram not in bigrams:
				bigrams[bigram] = Error(bigram, 'question')

			if true_label not in true_labels:
				true_labels[true_label] = Error(true_label, 'true_label')

			if guessed_label not in guessed_labels:
				guessed_labels[guessed_label] = Error(guessed_label, 'guessed_label')


			unigrams[unigram].incr_guessed_labels(guessed_label)
			unigrams[unigram].incr_true_labels(true_label)

			bigrams[bigram].incr_guessed_labels(guessed_label)
			bigrams[bigram].incr_true_labels(true_label)

			true_labels[true_label].incr_guessed_labels(guessed_label)
			true_labels[true_label].incr_question_unigrams(unigram)
			true_labels[true_label].incr_question_bigrams(bigram)

			guessed_labels[guessed_label].incr_true_labels(true_label)
			guessed_labels[guessed_label].incr_question_unigrams(unigram)
			guessed_labels[guessed_label].incr_question_bigrams(bigram)

			unigrams[unigram].incr_correct(is_correct)
			bigrams[bigram].incr_correct(is_correct)
			true_labels[true_label].incr_correct(is_correct)
			guessed_labels[guessed_label].incr_correct(is_correct)

	
	print("Total accuracy is " + str(float(num_correct) / num_tested * 100))


	REPORT_DIR = DATA_DIR + 'report/'

	try:
	    os.stat(REPORT_DIR)
	except:
	    os.mkdir(REPORT_DIR)


	fd = open(REPORT_DIR + 'unigrams.csv', 'w')

	for unigram in unigrams:
		unigram = unigrams[unigram]
		fd.write(str(unigram.value) + "~" + str(unigram.num_correct) + "/" + str(unigram.num_tested) + '\n')

	fd.close()
	fd = open(REPORT_DIR + 'bigrams.csv', 'w')
	for bigram in bigrams:
		bigram = bigrams[bigram]
		fd.write(str(bigram.value) + "~" + str(bigram.num_correct) + "/" + str(bigram.num_tested) + '\n')

	fd.close()

	fd = open(REPORT_DIR + 'guessed_labels.csv', 'w')
	for guessed_label in guessed_labels:
		guessed_label = guessed_labels[guessed_label]
		fd.write(str(guessed_label.value) + "~" + str(guessed_label.num_correct) + "/" + str(guessed_label.num_tested) + '\n')

	fd.close()

	fd = open(REPORT_DIR + 'true_labels.csv', 'w')
	for true_label in true_labels:
		true_label = true_labels[true_label]
		fd.write(str(true_label.value) + "~" + str(true_label.num_correct) + "/" + str(true_label.num_tested) + '\n')

	fd.close()

def compare(str1, str2):
	return str1.find(str2) > -1 or str2.find(str1) > -1


if __name__ == "__main__":
    # execute only if run as a script
    if (len(sys.argv) < 2):
    	print("Usage: python error_report <model-name>")
    	print("Usage: python error_report lstm")
    else:
		generate_report(sys.argv[1])