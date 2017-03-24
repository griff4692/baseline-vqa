import numpy as np
import os
import word_table as w_t

def serialize_errors(inputs, predictions, labels, model_name, wt, uncovered_train_labels, uncovered_test_labels):
	N = len(predictions)

	dirPath = './results/' + model_name;

	if not os.path.isdir(dirPath):
		os.makedirs(dirPath)

	np.save(dirPath + '/predictions', predictions)
	np.save(dirPath + '/labels', labels)
	np.save(dirPath + '/inputs', inputs)

	fd = open(dirPath + '/errors.csv', 'w');

	for i in range(N):
		prediction = predictions[i]
		prediction_idx = np.argmax(prediction)

		label = labels[i]
		label_idx = np.argmax(label)

		y_hat = wt.labels[prediction_idx]
		y = wt.labels[label_idx]

		question_embed = inputs[i]
		question = ''

		for i in range(10):
			idx = question_embed[i]
			word = wt.getWord(int(idx) - 1)

			question += word

			if i < 9 and not question_embed[i + 1] == 0:
				question += ' '
			else:
				break

		fd.write(question + '~' + y_hat + '~' + y + '\n')

	fd.close()

	fd = open(dirPath + '/uncovered_train.csv', 'w');

	for i in range(len(uncovered_train_labels)):
		fd.write(uncovered_train_labels[i] + "\n")

	fd.close()
	fd = open(dirPath + '/uncovered_test.csv', 'w');

	for i in range(len(uncovered_test_labels)):
		fd.write(uncovered_test_labels[i] + "\n")

	fd.close()

if __name__ == "__main__":
	dirPath = './results/lstm/';

    # execute only if run as a script
	inputs = np.load(dirPath + 'inputs.npy')
	predictions = np.load(dirPath + 'predictions.npy')
	labels = np.load(dirPath + 'labels.npy')

	wt = w_t.WordTable()
	wt.top_answers('data_processed.csv')
	serialize_errors(inputs, predictions, labels, 'lstm', wt)




