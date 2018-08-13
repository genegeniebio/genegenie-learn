'''
sbclearn (c) University of Manchester 2018

sbclearn is licensed under the MIT License.

To view a copy of this license, visit <http://opensource.org/licenses/MIT/>.

@author:  neilswainston
'''
# pylint: disable=invalid-name
# pylint: disable=unused-argument
import collections

from sklearn.base import BaseEstimator, TransformerMixin
from synbiochem.utils import seq_utils

import numpy as np


SEQ_IDX = 0

# KD Hydrophobicity, EIIP, Helix, Sheet, Turn
AA_PROPS = {
    'A': [1.8, -0.0667, 32.9, -23.6, -41.6],
    'R': [-4.5, 0.2674, 0, -6.2, -5.1],
    'N': [-3.5, -0.2589, -24.8, -41.6, 44.5],
    'D': [-3.5, 0.4408, 5.8, -41.6, 37.8],
    'C': [2.5, 0.1933, -5.1, 6.8, 17.4],
    'Q': [-3.5, 0.1545, 11.3, 0, -2],
    'E': [-3.5, -0.2463, 36.5, -67.3, -30.1],
    'G': [-0.4, -0.2509, -46.2, -13.9, 44.5],
    'H': [-3.2, -0.1414, 11.3, -18.6, -5.1],
    'I': [4.5, -0.2794, -1, 45.1, -75.5],
    'L': [3.8, -0.2794, 26.2, 15.7, -52.8],
    'K': [-3.9, -0.0679, 19.1, -31.5, 1],
    'M': [1.9, 0.1899, 27.8, 1, -51.1],
    'F': [2.8, 0.26, 10.4, 20.7, -51.1],
    'P': [-1.6, -0.1665, -59.8, -47.8, 41.9],
    'S': [-0.8, 0.1933, -32.9, -6.2, 35.8],
    'T': [-0.7, 0.2572, -24.8, 28.5, -4.1],
    'W': [-0.9, 0.0331, 3, 21.5, -4.1],
    'Y': [-1.3, 0.0148, -31.5, 27, 13.1],
    'V': [4.2, -0.2469, -3, 49.5, -69.3]
}

NUM_AA_PROPS = len(AA_PROPS['A'])


class OneHotTransformer(BaseEstimator, TransformerMixin):
    '''Transformer class to perform one-hot encoding on sequences.'''

    def __init__(self, nucl=True):
        alphabet = seq_utils.NUCLEOTIDES if nucl \
            else seq_utils.AA_CODES.values() + ['-', 'X']
        self.__alphabet = sorted(alphabet)

        # Define a mapping of chars to integers:
        self.__char_to_int = {c: i for i, c in enumerate(self.__alphabet)}

    def fit(self, *unused):
        '''fit.'''
        return self

    def transform(self, X, *unused):
        '''transform.'''
        return self.__one_hot_encode(X)

    def __one_hot_encode(self, X):
        '''One hot encode a DataFrame.'''
        encoded = [self.__one_hot_encode_seq(seq) for seq in X[:, SEQ_IDX]]

        return np.c_[X, encoded]

    def __one_hot_encode_seq(self, seq):
        '''One hot encode a seq.'''
        int_encoded = [self.__char_to_int[char] for char in seq]

        # One hot encode:
        one_hot_encoded = []

        for value in int_encoded:
            letter = [0 for _ in range(len(self.__alphabet))]
            letter[value] = 1
            one_hot_encoded.extend(letter)

        return one_hot_encoded


class AminoAcidTransformer(BaseEstimator, TransformerMixin):
    '''Transformer class to perform amino acid property encoding on
    sequences.'''

    def __init__(self, scale=(0.1, 0.9)):
        self.__scaled_aa_props = _scale(scale)
        self.__mean_values = [(scale[1] - scale[0]) / 2.0] * \
            len(self.__scaled_aa_props['A'])

    def fit(self, *unused):
        '''fit.'''
        return self

    def transform(self, X, *unused):
        '''transform.'''
        return self.__aa_encode(X)

    def __aa_encode(self, X):
        '''Amino acid property encode a DataFrame.'''
        encoded = [self.__aa_encode_seq(seq) for seq in X[:, SEQ_IDX]]

        return np.c_[X, encoded]

    def __aa_encode_seq(self, seq):
        '''Amino acid property encode a seq.'''
        encoded = [self.__scaled_aa_props.get(am_acid, self.__mean_values)
                   for am_acid in seq]

        return [val for sublist in encoded for val in sublist]


def _scale(scale):
    '''Scale amino acid properties.'''
    scaled = collections.defaultdict(list)

    for i in range(NUM_AA_PROPS):
        props = {key: value[i] for key, value in AA_PROPS.iteritems()}
        min_val, max_val = min(props.values()), max(props.values())

        for key, value in props.iteritems():
            scaled[key].append(scale[0] + (scale[1] - scale[0]) *
                               (value - min_val) / (max_val - min_val))

    return scaled
