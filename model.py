from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
import numpy as np


def build_model(corpus, val_indices, max_len, N_epochs=128):

    # pas encore trop certain de ça, ou de quoi que ce soit en fait
    timestep = 100000
    dims = 168

    # 2 hidden layer LSTM
    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape=(timestep, dims)))
    model.add(Dropout(0.2))
    model.add(LSTM(128, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(N_values))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    # le model.fit permet d'entrainer le model, sauf que je sais pas vraiment comment lui dire de s'entrainer sur tel ou tel trucs
    # model.fit(<<<TESTS SONGS>>>, batch_size=128, nb_epoch=N_epochs)

    return model