from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
import numpy as np


def build_model(corpus, max_len):
    # The in/output size is the same for each sample
    vectorSize = len(corpus[0][0])

    # 2 hidden layer LSTM
    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape=(max_len,
                                                            vectorSize)))
    model.add(Dropout(0.2))
    model.add(LSTM(128, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(vectorSize))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='rmsprop',
                  metrics=['accuracy'])

    return model


def train_model(model, callbacks_list, corpus, max_len, epochs, batch_size):
    X = []
    Y = []

    # Split input vector from output
    for music in corpus:
        if music is None:
            continue

        music_length = len(music)
        for i in range(0, music_length):
            predictionOffset = i + max_len
            if predictionOffset > music_length - 1:
                break

            X.append(music[i:predictionOffset])
            Y.append(music[predictionOffset])

    X = np.array(X)
    Y = np.array(Y)

    model.fit(X, Y, batch_size=batch_size, nb_epoch=epochs,
              validation_split=0.2, callbacks=callbacks_list)
