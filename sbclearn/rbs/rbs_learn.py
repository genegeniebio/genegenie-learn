'''
sbclearn (c) University of Manchester 2017

sbclearn is licensed under the MIT License.

To view a copy of this license, visit <http://opensource.org/licenses/MIT/>.

@author:  neilswainston
'''
# pylint: disable=invalid-name
import sys

import scipy.stats

from sbclearn.theanets import theanets_utils
import numpy as np
import pandas as pd
import sbclearn


def get_data(filename):
    '''Gets data.'''
    df = pd.read_table(filename)

    # TODO: Implement multiple sequence alignment,
    # TODO: deal with duplicate sequences (same seq, different vals)
    x_data = np.array(_encode_seqs(df['Sequence']))
    y_data = df['Output']
    labels = df['id']

    return x_data, y_data, labels


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
    x_data, y_data, _ = get_data(args[0])
    results = theanets_utils.k_fold_cross_valid((x_data, y_data))
    sbclearn.output(results)
    sbclearn.plot(results, 'Prediction of limonene production from RBS seqs')


if __name__ == '__main__':
    main(sys.argv[1:])
