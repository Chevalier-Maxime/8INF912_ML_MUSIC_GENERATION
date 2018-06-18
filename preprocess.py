#!/bin/env python
# -*- coding: utf-8 -*-

import sys
assert sys.version_info >= (3, 4)  # noqa: E402

import os
import logging
import numpy as np
import pickle
import gzip
from stats import compile_stats, stats, detect_time_complexity, \
                  display_time_step_graph
from netformat import convert_score_to_network_format,   \
                      convert_score_from_network_format, \
                      IncompatibleTimeSteps
from argparse import ArgumentParser
from shutil import which
from concurrent.futures import ProcessPoolExecutor, as_completed
from tempfile import gettempdir
from uuid import uuid4
from subprocess import run, DEVNULL, CalledProcessError
from pathlib import Path
from music21 import exceptions21, converter
from music21.interval import Interval
from music21.pitch import Pitch
from music21.chord import Chord
from music21.analysis.discrete import DiscreteAnalysisException
from tqdm import tqdm


# If Windows
if os.name == 'nt':
    MSCORE = 'MuseScore.exe'
else:
    MSCORE = 'mscore'

# Check mscore is accessible (so is installed too)
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
    data = {}
    process_args = ((cur_file, args) for cur_file in files)

    with ProcessPoolExecutor() as executor:
        for result in tqdm_parallel_map(executor, process_one, process_args):
            data.update(result)

    data = flatten(data)

    if args.stats:
        display_time_step_graph(data['denominator'])
        print(compile_stats(data))
    else:
        raw_data = np.array(data['raw_data'])
        output_file_loc = os.path.join(args.output_folder, 'raw.dat')
        with gzip.open(output_file_loc, 'wb') as f:
            raw_data = pickle.dump(raw_data, f)


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


def flatten(dic):
    res = {}

    for fileLoc in dic.keys():
        if dic[fileLoc] is None:
            continue

        for stat in dic[fileLoc].keys():
            if res.get(stat) is None:
                res[stat] = []

            var = dic[fileLoc][stat]
            res[stat].append(var)

    return res


def process_one(process_args):
    file_loc, args = process_args
    retDic = {file_loc: None}

    try:
        score = open_midi(file_loc)
        if args.stats:
            retDic[file_loc] = stats(score)
        else:
            raw_data = modify_piece(score, file_loc, args)
            retDic[file_loc] = {'raw_data': raw_data}

    except exceptions21.StreamException:
        logger.error('Cannot translate %s to music21 stream' % file_loc)
    except CalledProcessError:
        logger.error('mscore failed to convert %s from midi to musixcml'
                     % file_loc)

    return retDic


def modify_piece(score, file_loc, args):
    # Skip if too complex
    (denominator, time_max) = detect_time_complexity(score)
    if denominator > args.max_time_step_per_quarter:
        return

    if not args.keep_percussions:
        remove_percussions(score)

    if not args.keep_tonic:
        try:
            transpose_to_c_tonic(score)
        except DiscreteAnalysisException:
            logger.error('Unable to get key signature for %s' % file_loc)
            return

    if not args.keep_octaves:
        remove_bass(score)

    # Flatten (merge into 1 track)
    score = score.flat

    # Convert to network format
    try:
        raw_data = convert_score_to_network_format(score, args, time_max)
    except IncompatibleTimeSteps as e:
        logger.error('Cannot represent %s duration with %d time steps for %s' %
                     (str(e.time_step), args.max_time_step_per_quarter,
                      file_loc))
        return

    if args.debug_output_midi:
        score = convert_score_from_network_format(raw_data, args, time_max)
        save_midi(score, file_loc, args)

    return raw_data


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


def remove_percussions(score):
    # Remove the track containing percussions : midi channel 9 (from 0)
    # or 10 (from 1)
    for part in score.parts:
        if part.getInstrument().midiChannel == 9:
            score.remove(part)


def transpose_to_c_tonic(score):
    tonic = score.analyze('key').tonic
    transpos_interval = Interval(tonic, Pitch('C'))

    score.transpose(transpos_interval, inPlace=True)


def remove_bass(score):
    for note in score.recurse().notes:
        if isinstance(note, Chord):
            for chordNote in note:
                if chordNote.octave < 3:
                    chordNote.octave = 3

        elif note.octave < 3:
            note.octave = 3


def get_argument_parser():
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
    parser.add_argument('--keep-octaves', dest='keep_octaves',
                        action='store_true', help='Do not remove octave < 3')
    parser.add_argument('--stats', dest='stats', action='store_true',
                        help='Display some statistics')
    parser.add_argument('--max-time-step-per-quarter', type=int,
                        dest='max_time_step_per_quarter', action='store',
                        default=4, help='Skipping pieces requiring more than \
                        the specified value as time steps per quarter \
                        (default: %(default)s)')
    parser.add_argument('--debug-output-midi', dest='debug_output_midi',
                        action='store_true', help='Save midi files before \
                        translation to network format')

    return parser


def main(args):
    mid_files = scan_mid_files(args.input_folder)
    process_all(mid_files, args)


if __name__ == "__main__":
    args = (get_argument_parser()).parse_args()
    main(args)
