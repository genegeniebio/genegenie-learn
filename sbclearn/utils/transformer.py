'''
sbclearn (c) University of Manchester 2018

sbclearn is licensed under the MIT License.

To view a copy of this license, visit <http://opensource.org/licenses/MIT/>.

@author:  neilswainston
'''
# pylint: disable=invalid-name
# pylint: disable=unused-argument
from sklearn.base import BaseEstimator, TransformerMixin
from synbiochem.utils import seq_utils
import numpy as np

SEQ_IDX = 0


class OneHotTransformer(BaseEstimator, TransformerMixin):
    '''Transformer class to perform one-hot encoding on sequences.'''

    def __init__(self, nucl=True):
        alphabet = seq_utils.NUCLEOTIDES if nucl \
            else seq_utils.AA_CODES.values()
        self.__alphabet = sorted(alphabet)

        # Define a mapping of chars to integers:
        self.__char_to_int = {c: i for i, c in enumerate(self.__alphabet)}

    def fit(self, X, y=None):
        '''fit.'''
        return self

    def transform(self, X, y=None):
        '''transform.'''
        return self.__one_hot_encode(X)

    def __one_hot_encode(self, X):
        '''One hot encode a DataFrame.'''
        onehot_encoded = [self._one_hot_encode_seq(seq)
                          for seq in X[:, SEQ_IDX]]

        return np.c_[X, onehot_encoded]

    def _one_hot_encode_seq(self, seq):
        '''One hot encode a seq.'''
        int_encoded = [self.__char_to_int[char] for char in seq]

        # One hot encode:
        onehot_encoded = []

        for value in int_encoded:
            letter = [0 for _ in range(len(self.__alphabet))]
            letter[value] = 1
            onehot_encoded.extend(letter)

        return onehot_encoded
