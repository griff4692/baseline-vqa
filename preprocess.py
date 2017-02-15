import json
from collections import Counter

# open data file and parse
dataFileR = open('./data/data_filename.json', 'r')
datasetJSON = dataFileR.read()
dataset =  json.loads(datasetJSON)
dataFileR.close()

# Input: arrow of answers
# takes majority vote
def majorityAnswer(answers):
	answer, count = Counter(answers).most_common(1)[0]
	return answer

# open file for writing processed training examples
dataFileW = open('./data/data_processed.csv', 'w')

for imgId in dataset:
	questions = dataset[imgId]

	# each question counts as its own training example
	for question in questions:
		trainExampleStr = str(imgId) + ',' + str(question) + ',' + str(majorityAnswer(questions[question]))
		dataFileW.write(trainExampleStr + '\n')

print "finished processing data..."

# close up shop
dataFileW.close()