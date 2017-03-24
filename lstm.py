import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.callbacks import ModelCheckpoint
import json
import word_table as w_t
from keras.utils import np_utils
from error_analysis import serialize_errors

wt = w_t.WordTable()
wt.load_dictionary()
wt.top_answers('data_processed_val.csv')

import csv
import numpy as np
ques_maxlen = 20
X = []
Y = []
#mapping = {}
total = 0
uncovered_train_labels = []
with open('data_processed.csv', 'rb') as csvfile:
    dp = csv.reader(csvfile, delimiter='~')
    for row in dp:
        ques = row[1]
        ques = ques.lower().strip().strip('?!.').split()
        x = np.zeros(ques_maxlen)
        leng = len(ques)
        label = row[2]
        labelIdx = wt.getLabelIdx(label)

        if labelIdx > -1:
            Y.append(labelIdx)

            for i in range(ques_maxlen):
                if i < leng:
                    x[i] = wt.getIdx(ques[i])+1
            
            X.append(x)

        else:
            uncovered_train_labels.append(label)

X_data = np.array(X)
#X_data = np.reshape(X_data, (X_data.shape[0], X_data.shape[1], 1))
Y_data = np_utils.to_categorical(Y)

X = []
Y = []
uncovered_test_labels = []
with open('data_processed_val.csv', 'rb') as csvfile:
    dp = csv.reader(csvfile, delimiter='~')
    for row in dp:
        ques = row[1]
        ques = ques.lower().strip().strip('?!.').split()
        x = np.zeros(ques_maxlen)
        leng = len(ques)

        label = row[2]
        labelIdx = wt.getLabelIdx(label)

        if labelIdx > -1:
            Y.append(labelIdx)

            for i in range(ques_maxlen):
                if i < leng:
                    x[i] = wt.getIdx(ques[i])+1
    		
            X.append(x)

        else:
            uncovered_test_labels.append(label)

X_test = np.array(X)
#X_data = np.reshape(X_data, (X_data.shape[0], X_data.shape[1], 1))
Y_test = np_utils.to_categorical(Y)
    

top_words = len(wt.word2Idx)+1

# create the model
embedding_vector_length = 300
model = Sequential()
model.add(Embedding(top_words, embedding_vector_length, input_length=ques_maxlen))
model.add(LSTM(512, dropout_W = 0.2, dropout_U = 0.2))
model.add(Dense(Y_data.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

model.fit(X_data, Y_data, nb_epoch=10, batch_size = 64)
# Final evaluation of the model
scores, acc = model.evaluate(X_test, Y_test, verbose=1)
print acc

# generates errors by predicting on test set
predictions = model.predict(X_test)
serialize_errors(X_test, predictions, Y_test, 'lstm', wt, uncovered_train_labels, uncovered_test_labels)
