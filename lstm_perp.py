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

top_words = len(wt.word2Idx)+1
import csv
import numpy as np
ques_maxlen = 20
X = []
Y = []
#mapping = {}
total = 0
Xtrain_cont = [] 
Xtest_cont = []
ngram = 4


with open('data_processed.csv', 'rb') as csvfile:
    dp = csv.reader(csvfile, delimiter='~')
    for row in dp:
        ques = row[1]
        ques = ques.lower().strip().strip('?!.').split()
        leng = len(ques)
        try:
            for i in range(leng):
                Xtrain_cont.append(wt.getIdx(ques[i])+1)  
        except:
            pass

with open('data_processed_val.csv', 'rb') as csvfile:
    dp = csv.reader(csvfile, delimiter='~')
    for row in dp:
        ques = row[1]
        ques = ques.lower().strip().strip('?!.').split()
        leng = len(ques)
        try:
            for i in range(leng):
                Xtest_cont.append(wt.getIdx(ques[i])+1)  
        except:
            pass        

Xdata = []
Ydata = []
for i in range(len(Xtrain_cont) - ngram):
    Xdata.append(Xtrain_cont[i:i+ngram])
    Ydata.append(Xtrain_cont[i+ngram])

Xdata = np.array(Xdata)
Ydata = np_utils.to_categorical(Ydata)

Xtest = []
Ytest = []
for i in range(len(Xtest_cont) - ngram):
    Xtest.append(Xtest_cont[i:i+ngram])
    Ytest.append(Xtest_cont[i+ngram])

Xtest = np.array(Xtest)
Ytest = np_utils.to_categorical(Ytest, top_words)

    
# create the model
embedding_vecor_length = 64
model = Sequential()
model.add(Embedding(top_words, embedding_vecor_length, input_length=ngram))
model.add(LSTM(300, dropout_W = 0.2, dropout_U = 0.2))
model.add(Dense(top_words, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

model.fit(Xdata, Ydata, nb_epoch=10, batch_size = 64)

# Finding accuracy
scores, acc = model.evaluate(Xtest, Ytest, verbose=1)
print "Accuracy:"
print acc

#Finding perplexity
predictions = model.predict(Xtest)
prob = np.sum(np.multiply(Ytest, predictions), 1)
entropy = np.sum(np.multiply(np.log2(prob), prob))
print "Entropy"
print entropy
perplexity = (1.0/2**entropy)
print "Perplexity Final:"
print perplexity
# serialize_errors(X_test, predictions, Y_test, 'lstm', wt)
