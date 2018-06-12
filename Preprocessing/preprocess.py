#!/bin/env python
# -*- coding: utf-8 -*-

import sys
assert sys.version_info >= (3, 4)  # noqa: E402

import os
import logging
from argparse import ArgumentParser
from shutil import which
from math import inf
from fractions import Fraction
from concurrent.futures import ProcessPoolExecutor, as_completed
from tempfile import gettempdir
from uuid import uuid4
from subprocess import run, DEVNULL
from pathlib import Path
from music21 import interval, pitch, exceptions21, converter
from tqdm import tqdm


# Check mscore is accessible (so is installed too)
MSCORE = 'mscore'
assert which(MSCORE) is not None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('Preprocessor')


class EmptyDataFolderException(Exception):
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

    try:
        score = open_midi(file_loc)
    except exceptions21.StreamException:
        logger.error('Cannot translate %s to music21 stream' % file_loc)
        return
    except IndexError:
        logger.error('Known unknown error when translating %s to music21 \
                      stream' % file_loc)
        return

    if not args.keep_percussions:
        filter_out_percussions(score)

    if not args.keep_tonic:
        transpose_c(score)

    # TODO: The purpose of the next line is to check that preprocessing
    # operations are right
    save_midi(score, file_loc, args)

    # TODO: Convert to network format
    (denominator, longestNote) = detect_time_complexity(score)
    logger.info('%s: count units per quarter: %s' %
                (file_loc, str(denominator)))


def open_midi(file_loc):
    # A non-user defined name should prevent code injection
    temp_file = os.path.join(gettempdir(), str(uuid4()) + '.musicxml')
    run([MSCORE, '-o', temp_file, file_loc],
        check=True, stdout=DEVNULL, stderr=DEVNULL)

    score = converter.parse(temp_file)
    os.remove(temp_file)

    return score


def save_midi(score, file_loc, args):
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
    score.write('midi', output_file_loc)


def filter_out_percussions(score):
    # Remove the track containing percussions : midi channel 9 (from 0)
    # or 10 (from 1)
    for part in score.parts:
        if part.getInstrument().midiChannel == 9:
            score.remove(part)


def transpose_c(score):
    tonic = score.analyze('key').tonic
    transpos_interval = interval.Interval(tonic, pitch.Pitch('C'))

    score.transpose(transpos_interval, inPlace=True)


def detect_time_complexity(score):
    maxTime = -inf
    denominator = 1

    for curNote in score.recurse().notesAndRests:
        quarterLength = curNote.duration.quarterLength
        if quarterLength > maxTime:
            maxTime = quarterLength

        fraction = Fraction(1, denominator) + Fraction(quarterLength)
        if fraction.denominator > denominator:  # No simplification
            denominator = fraction.denominator

    return (denominator, maxTime)


if __name__ == "__main__":
    parser = ArgumentParser()

    # Mandatory arguments
    parser.add_argument('input_folder', type=str, help='The folder containing \
            the game folders containing *.mid (MIDI) files')
    parser.add_argument('output_folder', type=str, help='The folder that will \
            be provided to the network')

    # Optional arguments
    parser.add_argument('--keep-percussions', dest='keep_percussions',
                        action='store_true', help='Do not remove percussions')
    parser.add_argument('--keep-tonic', dest='keep_tonic', action='store_true',
                        help='Do not transpose to C')

    args = parser.parse_args()

    mid_files = scan_mid_files(args.input_folder)
    process_all(mid_files, args)
