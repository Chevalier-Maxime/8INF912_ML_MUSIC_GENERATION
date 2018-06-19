#!/bin/env python
# -*- coding: utf-8 -*-


import numpy as np
from music21 import tinyNotation
from argparse import ArgumentParser
from keras.models import model_from_json
from os import path, makedirs
from netformat import convert_score_to_network_format, \
                      convert_score_from_network_format
from stats import detect_time_complexity


def main():
    arg_parser = ArgumentParser()
    arg_parser.add_argument('input_folder', type=str, help='Path to the data \
                            whichs contains \'model.json\' and \'checkpoint\
                            .hdf5\'')
    arg_parser.add_argument('output_folder', type=str, help='Where to put \
                            generated musics')
    arg_parser.add_argument('--timesteps-per-quarter',
                            dest='timesteps_per_quarter', type=int,
                            help='Count of steps per quarter note',
                            default=4)
    arg_parser.add_argument('--sequence-length', dest='sequence_length',
                            type=int, help='Count of times (quarter note time \
                            like) for prediction', default=8)
    arg_parser.add_argument('--length', dest='length', type=int, help='Length \
                            (in quarter notes) of the musics to generate',
                            default=64)
    arg_parser.add_argument('--count', dest='count', type=int, help='Count \
                            of musics to generate', default=1)
    arg_parser.add_argument('--seed', dest='seed', type=str, help='First \
                            notes in tiny notation (\
                            http://web.mit.edu/music21/doc/moduleReference/moduleTinyNotation.html\
                            ) for the generated musics to begin with',
                            default='C D E F G A B C')
    args = arg_parser.parse_args()

    tspq = args.timesteps_per_quarter
    seq_len = args.sequence_length * tspq
    length_musics = args.length * tspq
    seed = tinyNotation.Converter(args.seed).parse().stream.flat
    count_musics = args.count
    input_folder = args.input_folder
    output_folder = args.output_folder

    # Setup output folder
    if not path.exists(output_folder):
        makedirs(output_folder)

    # Load the model
    with open(path.join(input_folder, 'model.json')) as f:
        model = model_from_json(f.read())

    # Load the weights
    model.load_weights(path.join(input_folder, 'checkpoint.hdf5'))

    # Convert seed to network format
    for i in range(0, count_musics):
        (denominator, max_time) = detect_time_complexity(seed)
        data = convert_score_to_network_format(seed, tspq, max_time)

        for j in range(0, length_musics):
            X = data[j:j+seq_len]
            X = np.expand_dims(X, axis=0)
            Y = model.predict(X)[0]
            data = np.vstack([data, Y])

        data = np.round(data)
        score = convert_score_from_network_format(data, tspq, 6)
        score.write('midi', path.join(output_folder, str(i) + '.mid'))

        seed.transpose(i + 1, inPlace=True)


if __name__ == "__main__":
    main()
