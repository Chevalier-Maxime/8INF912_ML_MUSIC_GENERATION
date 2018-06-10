#!/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
import argparse
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from music21 import midi, interval, pitch, exceptions21
from tqdm import tqdm


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('Preprocessor')


class EmptyDataFolderException(Exception):
    pass


class TranspositionException(Exception):
    pass


def scan_mid_files(data_folder):
    mid_files = []

    root_dir = os.listdir(data_folder)
    if len(root_dir) == 0:
        raise EmptyDataFolderException()

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
    process_args = ((cur_file, args) for cur_file in files)
    with ProcessPoolExecutor() as executor:
        for result in tqdm_parallel_map(executor, process_one, process_args):
            pass


# Thanks https://techoverflow.net/2017/05/18/
# how-to-use-concurrent-futures-map-with-a-tqdm-progress-bar/
def tqdm_parallel_map(executor, fn, *iterables, **kwargs):
    """
    Equivalent to executor.map(fn, *iterables),
    but displays a tqdm-based progress bar.

    Does not support timeout or chunksize as executor.submit is used internally

    **kwargs is passed to tqdm.
    """
    futures_list = []
    for iterable in iterables:
        futures_list += [executor.submit(fn, i) for i in iterable]
        for f in tqdm(as_completed(futures_list), total=len(futures_list),
                      **kwargs):
            yield f.result()


def process_one(process_args):
    file_loc, args = process_args

    piece = open_midi(file_loc)

    if not args.keep_percussions:
        filter_out_percussions(piece)

    if not args.keep_tonic:
        try:
            transpose_c(piece, file_loc)
        except TranspositionException as e:
            logger.error(e)
            return

    # TODO: Add below next preprocessing steps

    # TODO: The purpose of the next line is to check that preprocessing
    # operations are right
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


def transpose_c(piece, file_loc):
    try:
        score = midi.translate.midiFileToStream(piece)
    except exceptions21.StreamException:
        raise TranspositionException('Cannot translate ' + file_loc + ' to ' +
                                     'music21 stream')
    except IndexError:
        raise TranspositionException('Known unknown error when translating ' +
                                     file_loc + ' to music21 stream')

    tonic = score.analyze('key').tonic
    transpos_interval = interval.Interval(tonic, pitch.Pitch('C'))
    transpos_semitones = transpos_interval.semitones

    for track in piece.tracks:
        for event in track.events:
            if event.type == 'NOTE_ON' or event.type == 'NOTE_OFF':
                event.pitch += transpos_semitones


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
