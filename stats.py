#!/bin/env python
# -*- coding: utf-8 -*-


import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from math import inf
from fractions import Fraction
from music21.chord import Chord


def compile_stats(stats):
    return {'denominator': max(stats['denominator']),
            'longestNote': max(stats['longestNote']),
            'minOctave': min(stats['minOctave']),
            'maxOctave': max(stats['maxOctave'])}


def stats(score):
    (denominator, longestNote) = detect_time_complexity(score)
    (minOctave, maxOctave) = detect_min_max_octaves(score)

    dic = {'denominator': denominator,
           'longestNote': longestNote,
           'minOctave': minOctave,
           'maxOctave': maxOctave}

    return dic


def detect_time_complexity(score):
    maxTime = -inf
    denominator = 1

    for curNote in score.recurse().notesAndRests:
        quarterLength = curNote.quarterLength

        assert quarterLength

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


def update_min_max(minMaxVal, val):
    (minVal, maxVal) = minMaxVal

    if val < minVal:
        minVal = val
    elif val > maxVal:
        maxVal = val

    return (minVal, maxVal)
