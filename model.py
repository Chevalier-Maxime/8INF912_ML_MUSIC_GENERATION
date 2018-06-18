from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
import numpy as np


def build_model(corpus, max_len, epochs, batch_size):

    # The in/output size is the same for each sample
    vectorSize = len(corpus[0][0])

    # 2 hidden layer LSTM
    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape=(max_len, vectorSize)))
    model.add(Dropout(0.2))
    model.add(LSTM(128, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(vectorSize))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    # le model.fit permet d'entrainer le model, sauf que je sais pas vraiment comment lui dire de s'entrainer sur tel ou tel trucs
    # model.fit(<<<TESTS SONGS>>>, batch_size=128, nb_epoch=N_epochs)

    return model
