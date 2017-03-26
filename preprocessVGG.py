import numpy as np
from keras.applications.vgg16 import *
from keras.applications import vgg19
import argparse
from keras.models import Sequential, Model
from keras.preprocessing import image
from keras.layers import Dense, Dropout, Activation, Flatten, Input, merge, Merge
from keras.layers import Convolution2D, MaxPooling2D, LSTM, Embedding, GlobalAveragePooling2D
from keras.utils import np_utils
from keras import backend as K
from keras.preprocessing.sequence import pad_sequences
import os
import tensorflow as tf
import cPickle


base_img_model = VGG16(weights='imagenet', include_top=True, input_shape=(224,224,3))
vgg_features = Model(input=base_img_model.input, output = base_img_model.layers[21].output)

def getAndStore(args):
	total = 82782
	total_batches = total/args.batch_size + 1
	img_features = []
	filenames = []
	count = 0
	for image_file in os.listdir(args.image_dir):
		count+=1
		#print image_file
		filename = image_file.split(".")[0]
		img_vector = image.load_img(args.image_dir+image_file,target_size = (224,224))
		img_vector = image.img_to_array(img_vector)
		img_features.append(img_vector)
		filenames.append(filename)

		if(count%args.batch_size==0):
			print "Batch: %d out of %d batches"%(count/args.batch_size,total_batches)
			img_features = np.array(img_features)
			output_features = vgg_features.predict(img_features)
			for i in xrange(len(filenames)):
				np.save(args.output_dir+filenames[i], output_features[i])
			img_features = []
			filenames = []
	if(len(filenames)>0):
		print "final"
		img_features = np.array(img_features)
		output_features = vgg_features.predict(img_features)
		for i in xrange(len(filenames)):
			print output_features[i]
			#cPickle.dump(output_features[i], open(args.output_dir+filenames[i],'wb'))
			np.save(args.output_dir+filenames[i], output_features[i])
		img_features = []
		filenames = []


def main():
	parser = argparse.ArgumentParser(description='Image Precomputation')
	parser.add_argument('--batch_size', default=64, type=int, help='Batch VGGNet output')
	parser.add_argument('--image_dir', default='./data/training/')
	parser.add_argument('--output_dir', default='./data/sample_out/')

	args = parser.parse_args()
	args.img_dims = (224,224,3)

	return getAndStore(args)

if __name__ == '__main__':
	res = main()