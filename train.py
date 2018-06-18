#!/bin/env python
# -*- coding: utf-8 -*-


import gzip
import pickle
from argparse import ArgumentParser
from model import build_model, train_model
from keras.callbacks import ModelCheckpoint
from os import path, makedirs


def load_data(url):
    with gzip.open(url, 'rb') as f:
        data = pickle.load(f)

    return data


def main():
    arg_parser = ArgumentParser()
    arg_parser.add_argument('data_file', type=str, help='Path to the data \
                            file')
    arg_parser.add_argument('output_folder', type=str, help='Where to put the \
                            model and the learnt weights')
    arg_parser.add_argument('--timesteps-per-quarter',
                            dest='timesteps_per_quarter', type=int,
                            help='Count of LSTM steps per quarter note',
                            default=4)
    arg_parser.add_argument('--sequence-length', dest='sequence_length',
                            type=int, help='Count of times (quarter note time \
                            like) for prediction', default=8)
    arg_parser.add_argument('--epochs', dest='epochs', type=int, help='Stop \
                            the training when this value is reached',
                            default=25000)
    arg_parser.add_argument('--batch-size', dest='batch_size', type=str,
                            help='Size of batch required to update weights',
                            default=128)
    args = arg_parser.parse_args()

    corpus = load_data(args.data_file)
    max_len = args.sequence_length * args.timesteps_per_quarter
    epochs = args.epochs
    batch_size = args.batch_size
    output_folder = args.output_folder

    # Setup output folder
    if not path.exists(output_folder):
        makedirs(output_folder)

    # TODO: Resume function

    # Build and save model
    model = build_model(corpus, max_len)
    with open(path.join(args.output_folder, 'model.json'), 'w') as f:
        json_string = model.to_json()
        f.write(json_string)

    # Train and save weights
    checkpoint_path = path.join(args.output_folder, 'checkpoint.hdf5')
    callbacks_list = [ModelCheckpoint(checkpoint_path, monitor='val_acc',
                                      verbose=1, save_best_only=True,
                                      mode='max')]
    train_model(model, callbacks_list, corpus, max_len, epochs, batch_size)


if __name__ == "__main__":
    main()
