'''
sbclearn (c) University of Manchester 2017

sbclearn is licensed under the MIT License.

To view a copy of this license, visit <http://opensource.org/licenses/MIT/>.

@author:  neilswainston
'''
# pylint: disable=invalid-name
import os
import sys

from sklearn.ensemble.forest import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.tree.tree import DecisionTreeRegressor
from synbiochem.utils import seq_utils, xl_converter

import numpy as np
import pandas as pd
from sbclearn.utils import transformer


def get_data(xl_filename):
    '''Get data.'''
    dir_name = xl_converter.convert(xl_filename)
    dfs = []

    for dirpath, _, filenames in os.walk(dir_name):
        for filename in filenames:
            df = pd.read_csv(os.path.join(dirpath, filename))
            df['source'] = filename[:-4]
            dfs.append(df)

    df = pd.concat(dfs)
    df.set_index('id', inplace=True)
    df['seq'] = df['seq'].apply(lambda x: x.replace('*', ''))
    df['mutations'] = df['mutations'].apply(lambda x: '' if x == '[]' else x)

    return df.drop_duplicates()


def align(df, sources=None):
    '''Align.'''
    # Filter rows:
    if sources:
        df = df.loc[df['source'].isin(sources)]

    # Perform Clustal Omega alignment:
    df['align_seq'] = \
        pd.Series(seq_utils.do_clustal(df.to_dict()['seq']))

    # Strip out positions with identical residues:
    aas = [list(seq) for seq in df['align_seq']]

    char_df = pd.DataFrame(aas,
                           columns=['pos_' + str(val)
                                    for val in range(0, len(aas[0]))],
                           index=df.index)
    char_df = char_df.loc[:, (char_df != char_df.iloc[0]).any()]

    df['dif_align_seq'] = char_df.apply(''.join, axis=1)

    return df


def get_one_hot(df):
    '''Get one-hot encoding.'''
    trnsfrmr = transformer.OneHotTransformer(nucl=False)
    return trnsfrmr.transform(df.values)


def get_aa_props(df):
    '''Get amino acid property encoding.'''
    trnsfrmr = transformer.AminoAcidTransformer()
    return trnsfrmr.transform(df.values)


def estimate(X, y, cv):
    '''Estimate.'''
    print 'LinearRegression:\t' + \
        str(cross_valid_score(LinearRegression(), X, y, cv))
    print 'DecisionTreeRegressor:\t' + \
        str(cross_valid_score(DecisionTreeRegressor(), X, y, cv))
    print 'RandomForestRegressor:\t' + \
        str(cross_valid_score(RandomForestRegressor(), X, y, cv))


def cross_valid_score(estimator, X, y, cv, verbose=False):
    '''Perform cross validation.'''
    scores = cross_val_score(estimator, X, y, scoring='neg_mean_squared_error',
                             cv=cv,
                             verbose=verbose)
    scores = np.sqrt(-scores)

    return scores.mean(), scores.std()


def main(args):
    '''main method.'''
    df = get_data(args[0])
    df = align(df, args[1:] if len(args) > 1 else None)
    df.to_csv('out.csv')

    learn_df = df.loc[:, ['dif_align_seq', 'geraniol']]
    learn_df.columns = ['seq', 'activity']

    cv = 10

    encoded = get_one_hot(learn_df)
    estimate(encoded[:, 2:], encoded[:, 1], cv)

    encoded = get_aa_props(learn_df)
    estimate(encoded[:, 2:], encoded[:, 1], cv)


if __name__ == '__main__':
    main(sys.argv[1:])
