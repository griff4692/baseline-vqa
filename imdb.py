ngram = 4
from keras.datasets import imdb
import numpy as np
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

(X_train, y_train), (_, _) = imdb.load_data(path="imdb_full.pkl",
                                                      nb_words=None,
                                                      skip_top=0,
                                                      maxlen=None,
                                                      seed=113,
                                                      start_char=1,
                                                      oov_char=2,
                                                      index_from=3)

X_train = X_train

flatten = lambda l: [item for sublist in l for item in sublist]

X_train_cont = flatten(X_train)

Xdata = []
Ydata = []

for i in range(len(X_train_cont) - ngram):
    Xdata.append(X_train_cont[i:i+ngram])
    Ydata.append(X_train_cont[i+ngram])

top_words = np.max(np.array(Ydata));
ngram = 4
embedding_vecor_length = 64
model = Sequential()
model.add(Embedding(top_words, embedding_vecor_length, input_length=ngram))
model.add(LSTM(100, dropout_W = 0.2, dropout_U = 0.2))
model.add(Dense(top_words, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())


samples = 2000

def generate_data():
    while 1:
        for i in range(len(Xdata)/samples):
            xdata = np.array(Xdata[i*samples:(i+1)*samples])
            ydata = np_utils.to_categorical(Ydata[i*samples:(i+1)*samples], top_words)
            print i*samples,(i+1)*samples
            yield xdata, ydata

model.fit_generator(generate_data(), samples_per_epoch = len(Xdata) - samples, nb_epoch=5, verbose=2)
model.save("imdb_lstm.h5")
