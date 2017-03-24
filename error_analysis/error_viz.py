from collections import defaultdict
import os
import sys
import csv

def generate_report(model_name):
	REPORT_DIR = './results/' + model_name +'/report/'
	VIZ_DIR = './results/' + model_name +'/viz/'

	try:
	    os.stat(VIZ_DIR)
	except:
	    os.mkdir(VIZ_DIR)


	bigrams_count = defaultdict(int)
	bigrams_correct = defaultdict(float)
	bigrams_acc = defaultdict(float)

	unigrams_count = defaultdict(int)
	unigrams_correct = defaultdict(float)
	unigrams_acc = defaultdict(float)

	true_labels_count = defaultdict(int)
	true_labels_correct = defaultdict(float)
	true_labels_acc = defaultdict(float)

	with open(REPORT_DIR + 'bigrams.csv', 'rb') as csvfile:
		rows = csv.reader(csvfile, delimiter='~')

		for row in rows:
			bigram = row[0]

			fract = row[1].split('/')
			num = int(fract[0])
			denom = int(fract[1])

			acc = round(float(num)/denom*100)

			bigrams_count[bigram] = denom
			bigrams_correct[bigram] = num
			bigrams_acc[bigram] = acc


	with open(REPORT_DIR + 'unigrams.csv', 'rb') as csvfile:
		rows = csv.reader(csvfile, delimiter='~')

		for row in rows:
			unigram = row[0]
			fract = row[1].split('/')
			num = int(fract[0])
			denom = int(fract[1])

			acc = round(float(num)/denom*100)

			unigrams_count[unigram] = denom
			unigrams_correct[unigram] = num
			unigrams_acc[unigram] = acc


	numbers_correct = 0
	numbers_tested = 0

	with open(REPORT_DIR + 'true_labels.csv', 'rb') as csvfile:
		rows = csv.reader(csvfile, delimiter='~')

		for row in rows:
			true_label = row[0]
			fract = row[1].split('/')
			num = int(fract[0])
			denom = int(fract[1])

			acc = round(float(num)/denom*100)

			try:
			 	label_int = int(true_label)
				numbers_correct += num
				numbers_tested += denom
			except ValueError:
				pass

			true_labels_count[true_label] = denom
			true_labels_correct[true_label] = num
			true_labels_acc[true_label] = acc

	sorted_unigrams = sorted(unigrams_count, key=unigrams_count.get, reverse=True)
	sorted_bigrams = sorted(bigrams_count, key=bigrams_count.get, reverse=True)

	sorted_true_labels = sorted(true_labels_count, key=true_labels_count.get, reverse=True)


	print("Accuracy on digits is " + str(round(float(numbers_correct) / numbers_tested * 100)) + "%")

	k = 10

	top_unigrams = sorted_unigrams[0:k]
	top_bigrams = sorted_bigrams[0:k]
	top_true_labels = sorted_true_labels[0:k]

	other_unigrams = [0, 0]
	other_bigrams = [0, 0]
	other_true_labels = [0,0]

	for i in range(k, len(sorted_unigrams)):
		other_unigrams[0] += unigrams_correct[sorted_unigrams[i]]
		other_unigrams[1] += unigrams_count[sorted_unigrams[i]]

	for i in range(k, len(sorted_bigrams)):
		other_bigrams[0] += bigrams_correct[sorted_bigrams[i]]
		other_bigrams[1] += bigrams_count[sorted_bigrams[i]]

	for i in range(k, len(sorted_true_labels)):
		other_true_labels[0] += true_labels_correct[sorted_true_labels[i]]
		other_true_labels[1] += true_labels_count[sorted_true_labels[i]]

	fd = open(VIZ_DIR + 'unigrams', 'w')

	for unigram in top_unigrams:
		fd.write(unigram + '\t' + str(unigrams_acc[unigram]) + '\n')

	fd.write('Other' + '\t' + str(round(float(other_unigrams[0]) / other_unigrams[1] * 100)) + '\n')

	fd.close()

	fd = open(VIZ_DIR + 'bigrams', 'w')

	for bigram in top_bigrams:
		fd.write(bigram + '\t' + str(bigrams_acc[bigram]) + '\n')

	fd.write('Other' + '\t' + str(round(float(other_bigrams[0]) / other_bigrams[1] * 100)) + '\n')

	fd.close()

	fd = open(VIZ_DIR + 'true_labels', 'w')

	for true_label in top_true_labels:
		fd.write(true_label + '\t' + str(true_labels_acc[true_label]) + '\n')

	fd.write('Other' + '\t' + str(round(float(other_true_labels[0]) / other_true_labels[1] * 100)) + '\n')

	fd.close()



if __name__ == "__main__":
    # execute only if run as a script
    if (len(sys.argv) < 2):
    	print("Usage: python error_report <model-name>")
    	print("Usage: python error_report lstm")
    else:
		generate_report(sys.argv[1])