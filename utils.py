import os
import numpy as np
import pickle

# maps words to their embeddings
def generate_embedding_idx(embedding_dir, dim):
	# code borrowed from keras blog post
	# https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html
	embedding_idx = {}
	f = open(os.path.join(embedding_dir, 'glove.6B.%dd.txt' % dim))
	for line in f:
	    values = line.split()
	    word = values[0]
	    coefs = np.asarray(values[1:], dtype='float32')
	    embedding_idx[word] = coefs
	f.close()

	print('Found %s word vectors.' % len(embedding_idx))

	return embedding_idx

def serialize_ans_embedding_matrix(classDict, embedding_dir, dim):
	embedding_matrix = np.zeros([len(classDict), dim])
	embedding_idx = generate_embedding_idx(embedding_dir, dim)

	insert_idx = 0

	for word in classDict:
		embedding_vector = embedding_idx.get(word)
		if embedding_vector is not None:
			v_sq = np.square(embedding_vector)
			summed = np.sum(v_sq)
			summed_norm = np.sqrt(summed)
			embedding_matrix[insert_idx] = embedding_vector/np.sum(summed_norm)
			insert_idx += 1

	non_null_mat = embedding_matrix[:insert_idx,:]

	fd = open('./data/top_answer_embeddings', 'w')
	pickle.dump(non_null_mat, fd)
	fd.close()

def generate_embedding_matrix(wt, embedding_dir, dim):
	embedding_matrix = np.zeros([wt.vocabSize() + 1, dim])
	embedding_idx = generate_embedding_idx(embedding_dir, dim)

	not_found = 0

	for i in range(wt.vocabSize()):
		word = wt.getWord(i)
		embedding_vector = embedding_idx.get(word)
		if embedding_vector is not None:
			# words not found in embedding index will be all-zeros.
			embedding_matrix[i] = embedding_vector
		else:
			not_found += 1

	return embedding_matrix

def argrender(args):
	print("")
	for arg in vars(args):
		val = str(getattr(args, arg)) if not arg == 'ClassDict' else 'Dictionary of size ' + str(len(getattr(args, arg)))
		print(arg + '=' + val)
	print("")

def lossrender(args, iteration, metrics):
	string = "%d: Loss(%s)=%f\t" % (iteration, args.loss, metrics[0])
	for i in range(1, len(metrics)):
		string += "%s=%f\t" % (args.METRICS[i - 1], metrics[i])

	print(string)


	