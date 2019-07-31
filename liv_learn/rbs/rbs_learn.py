'''
(c) University of Liverpool 2019

All rights reserved.

@author: neilswainston
'''
# pylint: disable=invalid-name
# pylint: disable=ungrouped-imports
# pylint: disable=wrong-import-order
import sys

import liv_learn
from liv_learn.utils.validator import k_fold_cross_valid
import numpy as np
import pandas as pd


def get_data(filename):
    '''Gets data.'''
    df = pd.read_table(filename)

    # TODO: Implement multiple sequence alignment,
    # TODO: deal with duplicate sequences (same seq, different vals)
    return np.array(_encode_seqs(df['Sequence'])), df['Output']


def _encode_seqs(seqs):
    '''Encodes sequences.'''
    stripped_seqs = []

    for pos in zip(*seqs.tolist()):
        if len(set(pos)) > 1:
            stripped_seqs.append(pos)

    stripped_seqs = [''.join(nucls) for nucls in zip(*stripped_seqs)]

    encoding = {'A': [1, 0, 0, 0],
                'C': [0, 1, 0, 0],
                'G': [0, 0, 1, 0],
                'T': [0, 0, 0, 0],
                '-': [0.25, 0.25, 0.25, 0.25]}

    return [[item for sublist in [encoding[nucl] for nucl in seq]
             for item in sublist]
            for seq in stripped_seqs]


def main(args):
    '''main method.'''
    x_data, y_data = get_data(args[0])
    results = k_fold_cross_valid((x_data, y_data))
    liv_learn.plot(results, 'Prediction of limonene production from RBS seqs')


if __name__ == '__main__':
    main(sys.argv[1:])
