import json
from collections import Counter

csv_delimiter = '~'
default_data_path = './data_json/val2014.json'
def get_default_data_path():
	return default_data_path

def get_default_processed_data_path():
	return "./data_json/train2014.csv"

def generate_processed_path(data_filename):
	data_dir = data_filename[0:find_last_idx('/', data_filename) + 1]
	return data_dir + 'data_processed.csv'

def find_last_idx(char,str):
	pos = []
	str_len = len(str)
	for i in range(str_len):
		if char == str[i]:
			pos.append(i)

	return pos[-1]


# takes in file path to json data
# parses it in csv, comma-delimited format
# and saves to same directory under
# the name 'data_processed.csv'
def process_json(data_filename=default_data_path):
	# open data file and parse
	dataFileR = open(data_filename, 'r')
	datasetJSON = dataFileR.read()
	dataset =  json.loads(datasetJSON)
	dataFileR.close()

	# Input: arrow of answers
	# takes majority vote
	def majorityAnswer(answers):
		answer, count = Counter(answers).most_common(1)[0]
		return answer

	processed_data_file = generate_processed_path(data_filename);
	# open file for writing processed training examples
	dataFileW = open(processed_data_file, 'w')

	for imgId in dataset:
		questions = dataset[imgId]

		# each question counts as its own training example
		for question in questions:
			trainExampleStr = str(imgId) + csv_delimiter + str(question) + csv_delimiter + str(majorityAnswer(questions[question]))
			dataFileW.write(trainExampleStr + '\n')

	print "finished processing data..."

	# close up shop
	dataFileW.close()

	return processed_data_file


if __name__ == "__main__":
    # execute only if run as a script
    process_json()