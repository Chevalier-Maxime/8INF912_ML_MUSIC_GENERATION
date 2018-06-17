#!/bin/env python
# -*- coding: utf-8 -*-

import sys
assert sys.version_info >= (3, 4)  # noqa: E402

import os
import logging
import matplotlib.pyplot as plt
import numpy as np
import numbers
import pickle
import gzip
from math import ceil, floor
from matplotlib.ticker import PercentFormatter
from argparse import ArgumentParser
from shutil import which
from math import inf
from fractions import Fraction
from concurrent.futures import ProcessPoolExecutor, as_completed
from tempfile import gettempdir
from uuid import uuid4
from subprocess import run, DEVNULL, CalledProcessError
from pathlib import Path
from music21 import exceptions21, converter
from music21.interval import Interval, notesToChromatic
from music21.pitch import Pitch
from music21.chord import Chord
from music21.note import Note
from music21.stream import Stream
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


class IncompatibleTimeSteps(Exception):
    def __init__(self, tried_time_step):
        self.time_step = tried_time_step


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
            if result:
                data.update(result)

    data = flatten(data)

    if args.stats:
        display_time_step_graph(data['denominator'])
        print(compile_stats(stats))
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
        for stat in dic[fileLoc].keys():
            if res.get(stat) is None:
                res[stat] = []

            var = dic[fileLoc][stat]
            res[stat].append(var)

    return res


def compile_stats(stats):
    return {'denominator': max(stats['denominator']),
            'longestNote': max(stats['longestNote']),
            'minOctave': min(stats['minOctave']),
            'maxOctave': max(stats['maxOctave'])}


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
    finally:
        return retDic


def stats(score):
    (denominator, longestNote) = detect_time_complexity(score)
    (minOctave, maxOctave) = detect_min_max_octaves(score)

    dic = {'denominator': denominator,
           'longestNote': longestNote,
           'minOctave': minOctave,
           'maxOctave': maxOctave}

    return dic


def modify_piece(score, file_loc, args):
    # Skip if too complex
    (denominator, timeMax) = detect_time_complexity(score)
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
        raw_data = convert_score_to_network_format(score, args)
    except IncompatibleTimeSteps as e:
        logger.error('Cannot represent %s duration with %d time steps for %s' %
                     (str(e.time_step), args.max_time_step_per_quarter,
                      file_loc))
        return

    if args.debug_output_midi:
        score = convert_score_from_network_format(raw_data, args)
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


def convert_score_to_network_format(score, args):
    """ score must be flatten """

    lengthScore = ceil(time_to_timesteps(score, args)) + 1  # To add OFF event
    lengthInOutput = 7 * 12 * 2

    # First dim : One time step
    # Second dim : 7 octaves * 12 notes * 2 events (on, off)
    data = np.zeros((lengthScore, lengthInOutput))

    for note in score.recurse().notes:
        if isinstance(note, Chord):
            for subNote in note:
                add_note(data, subNote, args)

        else:
            add_note(data, note, args)

    return data


def add_note(data, note, args):
    offsetTimeNoteOn = offset_time(note, args)
    offsetPitchNoteOn = offset_pitch_on(note)

    offsetTimeNoteOff = offset_time_end(note, args)
    offsetPitchNoteOff = offset_pitch_off(note)

    # For consistency each note on must be followed by a note off
    # But an issue, can appear. What if:
    # 1 - The same note happen in the same time and don't have the same
    #     length?
    # 2 - The current note happen between another same note on/off events?
    # A note on event must be triggered at the end of the first note played
    # Examples (S : Start/Note ON ; E : End/Note OFF) :
    # _S_S____ must become  _S_S____
    # _____E_E              ___E___E

    # Note ON always applies
    data[offsetTimeNoteOn][offsetPitchNoteOn] = True

    # Detect conflict
    offsetTimeOtherNoteOff = get_time_end(data, offsetTimeNoteOn,
                                          offsetPitchNoteOff, args)
    if not offsetTimeOtherNoteOff:
        data[offsetTimeNoteOff][offsetPitchNoteOff] = True
    else:
        data[offsetTimeNoteOn][offsetPitchNoteOff] = True

        offsetTimeConflictEndNoteOff = max(offsetTimeNoteOff,
                                           offsetTimeOtherNoteOff)
        if offsetTimeOtherNoteOff != offsetTimeConflictEndNoteOff:
            data[offsetTimeOtherNoteOff][offsetPitchNoteOff] = False
            data[offsetTimeConflictEndNoteOff][offsetPitchNoteOff] = True


def convert_score_from_network_format(raw_data, args):
    score = Stream()

    for timestep in range(0, len(raw_data)):
        for event in range(0, len(raw_data[timestep])):
            if raw_data[timestep][event] == 1 and offset_pitch_is_on(event):
                pitch = offset_to_pitch(event)
                offset_off = offset_pitch_on_to_off(event)

                offset_time = timesteps_to_time(timestep, args)
                offset_timestep_end = get_time_end(raw_data, event,
                                                   offset_off, args)
                if not offset_timestep_end:
                    logger.warning('End event missing. Skipping note...')
                    continue

                length_timestep = offset_timestep_end - timestep
                length = timesteps_to_time(length_timestep, args)

                note = Note(pitch)
                note.offset = offset_time
                note.quarterLength = length

                score.insert(note, ignoreSort=True)

    score.sort()
    return score


def get_time_end(data, offsetTimeNoteOn, offsetPitchNoteOff, args):
    # Prevent for going beyond 6 quarter notes are the end of the score
    max_timesteps = min(offsetTimeNoteOn + time_to_timesteps(6, args),
                        len(data))

    # TODO: For now, we suppose 6 the maximum note length
    for curOffset in range(offsetTimeNoteOn, max_timesteps):
        if data[curOffset][offsetPitchNoteOff] == 1:
            return curOffset

    return None


def time_to_timesteps(obj, args):
    if isinstance(obj, numbers.Real):
        quarterLength = obj
    else:
        quarterLength = obj.quarterLength

    length = quarterLength * args.max_time_step_per_quarter

    if length % 1 != 0.0:  # if length is decimal, there is an issue
        raise IncompatibleTimeSteps(length)

    return int(length)


def timesteps_to_time(timesteps, args):
    return timesteps / args.max_time_step_per_quarter


def offset_time(note, args):
    offset = note.offset * args.max_time_step_per_quarter

    if offset % 1 != 0.0:  # If offset is decimal, there is an issue
        raise IncompatibleTimeSteps(offset)

    return int(offset)


def offset_time_end(note, args):
    return offset_time(note, args) + time_to_timesteps(note, args)


def offset_pitch_on(note):
    """ Note ON are pair offsets """
    return offset_octave(note) + offset_in_octave(note)


def offset_pitch_is_on(offset):
    return (offset % 2) == 0


def offset_pitch_off(note):
    """ Note OFF are impair offsets """
    return offset_octave(note) + offset_in_octave(note) + 1


def offset_pitch_is_off(offset):
    return (offset % 2) == 1


def offset_pitch_on_to_off(offset):
    return offset + 1


def offset_to_pitch(offset):
    p = Pitch('C3')
    return p.transpose(floor(offset / 2))


def offset_octave(note):
    return (note.octave - 3) * 24


def offset_in_octave(note):
    return semitones(note) * 2


def semitones(note):
    refPitch = Pitch('C3')
    return notesToChromatic(refPitch, note.pitch).semitones % 12


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


def detect_min_max_octaves(score):
    minMaxOctaves = (inf, -inf)

    for curNote in score.recurse().notes:
        if isinstance(curNote, Chord):
            for curChordNote in curNote:
                minMaxOctaves = update_min_max(minMaxOctaves,
                                               curChordNote.octave)
        else:
            minMaxOctaves = update_min_max(minMaxOctaves, curNote.octave)

    return minMaxOctaves


def update_min_max(minMaxVal, val):
    (minVal, maxVal) = minMaxVal

    if val < minVal:
        minVal = val
    elif val > maxVal:
        maxVal = val

    return (minVal, maxVal)


def display_time_step_graph(time_steps):
    countData = len(time_steps)

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax2.yaxis.set_major_formatter(PercentFormatter())

    def to_percent(ax1):
        y1, y2 = ax1.get_ylim()
        ax2.set_ylim(y1 / countData * 100, y2 / countData * 100)

    ax1.callbacks.connect('ylim_changed', to_percent)

    ax1.hist(time_steps, bins=max(time_steps), cumulative=True,
             histtype='step')
    ax1.set_xscale('log')
    ax1.set_title('Cumulated count of scores (total: ' + str(countData) + ') \
                   according to time steps')
    ax1.set_xlabel('Required time steps per 1/4 note')
    ax1.set_ylabel('Count of scores')
    ax2.set_ylabel('Relative count of scores')
    ax2.grid(True)

    plt.show()


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
