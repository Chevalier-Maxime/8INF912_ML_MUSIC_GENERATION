#!/bin/env python
# -*- coding: utf-8 -*-


import numbers
import logging
import numpy as np
from math import ceil, floor
from music21.chord import Chord
from music21.stream import Stream
from music21.note import Note
from music21.pitch import Pitch
from music21.interval import notesToChromatic


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('Preprocessor')


class IncompatibleTimeSteps(Exception):
    def __init__(self, tried_time_step):
        self.time_step = tried_time_step


def convert_score_to_network_format(score, args, time_max):
    """ score must be flatten """

    lengthScore = ceil(time_to_timesteps(score, args)) + 1  # To add OFF event
    lengthInOutput = 7 * 12 * 2

    # First dim : One time step
    # Second dim : 7 octaves * 12 notes * 2 events (on, off)
    data = np.zeros((lengthScore, lengthInOutput))

    for note in score.recurse().notes:
        if isinstance(note, Chord):
            for subNote in note:
                add_note(data, subNote, args, time_max)

        else:
            add_note(data, note, args, time_max)

    return data


def add_note(data, note, args, time_max):
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
                                          offsetPitchNoteOff, args, time_max)
    if not offsetTimeOtherNoteOff:
        data[offsetTimeNoteOff][offsetPitchNoteOff] = True
    else:
        data[offsetTimeNoteOn][offsetPitchNoteOff] = True

        offsetTimeConflictEndNoteOff = max(offsetTimeNoteOff,
                                           offsetTimeOtherNoteOff)
        if offsetTimeOtherNoteOff != offsetTimeConflictEndNoteOff:
            data[offsetTimeOtherNoteOff][offsetPitchNoteOff] = False
            data[offsetTimeConflictEndNoteOff][offsetPitchNoteOff] = True


def convert_score_from_network_format(raw_data, args, time_max):
    score = Stream()

    for timestep in range(0, len(raw_data)):
        for event in range(0, len(raw_data[timestep])):
            if raw_data[timestep][event] == 1 and offset_pitch_is_on(event):
                pitch = offset_to_pitch(event)
                offset_off = offset_pitch_on_to_off(event)

                offset_time = timesteps_to_time(timestep, args)
                offset_timestep_end = get_time_end(raw_data, timestep,
                                                   offset_off, args, time_max)
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


def get_time_end(data, offsetTimeNoteOn, offsetPitchNoteOff, args, time_max):
    # Prevent for going beyond 6 quarter notes are the end of the score
    max_timesteps = min(offsetTimeNoteOn + time_to_timesteps(time_max, args),
                        len(data))

    offsetTimeNoteOn += 1
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
