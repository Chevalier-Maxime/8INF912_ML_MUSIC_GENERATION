#!/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
import argparse
import logging
from pathlib import Path
from music21 import midi, interval, pitch, exceptions21


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('Preprocessor')


class EmptyDataFolder(Exception):
    pass


def scan_mid_files(data_folder):
    mid_files = []

    root_dir = os.listdir(data_folder)
    if len(root_dir) == 0:
        raise EmptyDataFolder()

    for node in root_dir:
        node_absolute = os.path.join(data_folder, node)
        if os.path.isdir(node_absolute):
            game_dir = os.listdir(node_absolute)

            if len(game_dir) == 0:
                logger.warning('No files in %s' % node_absolute)

            for node2 in game_dir:
                # Check extension
                ext = os.path.splitext(node2)[1]
                node2_absolute = os.path.join(node_absolute, node2)

                if ext.lower() == '.mid':
                    mid_files.append(node2_absolute)
                else:
                    logging.warning('%s is not a *.mid file' % node2_absolute)
        else:
            logger.warning('%s should be a folder' % node_absolute)

    return mid_files


def process_all(files, args):
    for f in files:
        process_one(f, args)


def process_one(file_loc, args):
    piece = open_midi(file_loc)

    if not args.keep_percussions:
        filter_out_percussions(piece)

    try:
        score = midi.translate.midiFileToStream(piece)
    except exceptions21.StreamException:
        logger.error('No tracks in %s' % file_loc)
        return

    if not args.keep_tonic:
        transpose_c(score)

    # TODO: Add below next preprocessing steps

    # TODO: The purpose of the next line is to check that preprocessing
    # operations are right
    piece = midi.translate.streamToMidiFile(score)
    save_midi(piece, file_loc, args)

    # TODO: Convert to network format


def open_midi(file_loc):
    piece = midi.MidiFile()
    piece.open(file_loc)
    piece.read()
    piece.close()

    return piece


def save_midi(piece, file_loc, args):
    """ if file_loc is '../data/game/mymusic.mid'. The file will be saved in
    <output_folder>/game/mymusic_melody.mid """

    # Extract folders name
    file_loc_norm = os.path.normpath(file_loc)
    file_loc_split = file_loc_norm.split(os.sep)

    # file_loc_split[-2] is the game folder
    folder = os.path.join(args.output_folder, file_loc_split[-2])

    # Create parents folders
    Path(folder).mkdir(parents=True, exist_ok=True)

    # file_loc_split[-1] is the music file
    file_parts = os.path.splitext(file_loc_split[-1])

    # Append _melody to the output file
    output_file_name = file_parts[0] + '_melody' + file_parts[1]

    # Finalize the path
    output_file_loc = os.path.join(folder, output_file_name)

    # Write
    piece.open(output_file_loc, attrib='wb')
    piece.write()
    piece.close()


def filter_out_percussions(piece):
    # Keep only tracks that do not contain the channel 10 (percussion)
    piece.tracks[:] = [t for t in piece.tracks if 10 not in t.getChannels()]


def transpose_c(score):
    tonic = score.analyze('key').tonic
    transpos_interval = interval.Interval(tonic, pitch.Pitch('C'))
    score.transpose(transpos_interval, inPlace=True)


if __name__ == "__main__":
    assert sys.version_info >= (3, 4)

    parser = argparse.ArgumentParser()
    parser.add_argument('input_folder', type=str, help='The folder containing \
            the game folders containing *.mid (MIDI) files')
    parser.add_argument('output_folder', type=str, help='The folder that will \
            be provided to the network')

    parser.add_argument('--keep-percussions', dest='keep_percussions',
                        action='store_true', help='Do not remove percussions')
    parser.add_argument('--keep-tonic', dest='keep_tonic', action='store_true',
                        help='Do not transpose to C')
    parser.set_defaults(keep_percussions=False)

    args = parser.parse_args()

    mid_files = scan_mid_files(args.input_folder)
    process_all(mid_files, args)
