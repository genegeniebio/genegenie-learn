'''
sbclearn (c) University of Manchester 2018

sbclearn is licensed under the MIT License.

To view a copy of this license, visit <http://opensource.org/licenses/MIT/>.

@author:  neilswainston
'''
# pylint: disable=invalid-name
import itertools
import os
import sys

from sklearn.ensemble.forest import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing.data import StandardScaler
from sklearn.svm import SVR
from sklearn.tree.tree import DecisionTreeRegressor
from synbiochem.utils import xl_converter

import numpy as np
import pandas as pd
from sbclearn.utils import aligner, transformer


def get_data(xl_filename, sources=None):
    '''Get data.'''
    df = _get_raw_data(xl_filename)
    df = aligner.align(df, sources)
    df.to_csv('out.csv')

    learn_df = df.loc[:, ['dif_align_seq', 'geraniol']]
    learn_df.columns = ['seq', 'activity']

    return learn_df.values


def _get_raw_data(xl_filename):
    '''Get raw data.'''
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


def cross_valid_score(estimator, X, y, cv, verbose=False):
    '''Perform cross validation.'''
    scores = cross_val_score(estimator,
                             X, y,
                             scoring='neg_mean_squared_error',
                             cv=cv,
                             verbose=verbose)
    scores = np.sqrt(-scores)

    return scores.mean(), scores.std()


def do_grid_search(estimator, param_grid, X, y):
    '''Perform grid search.'''
    grid_search = GridSearchCV(estimator,
                               param_grid,
                               scoring='neg_mean_squared_error',
                               cv=10,
                               verbose=True)

    grid_search.fit(X, y)

    res = grid_search.cv_results_

    for mean, params in sorted(zip(res['mean_test_score'], res['params']),
                               reverse=True):
        print (np.sqrt(-mean), params)

    print


def main(args):
    '''main method.'''
    print SVR(kernel='poly')

    data = get_data(args[0], args[1:] if len(args) > 1 else None)

    transformers = [transformer.OneHotTransformer(nucl=False),
                    transformer.AminoAcidTransformer()]

    estimators = [
        LinearRegression(),
        DecisionTreeRegressor(),
        RandomForestRegressor(),
        SVR(kernel='poly')]

    for trnsfrmr, estimator in itertools.product(transformers, estimators):
        encoded = trnsfrmr.transform(data)

        mean, std = cross_valid_score(estimator,
                                      encoded[:, 2:], encoded[:, 1],
                                      cv=10)

        print '\t'.join([trnsfrmr.__class__.__name__,
                         estimator.__class__.__name__,
                         str((mean, std))])

    print

    X, y = encoded[:, 2:], encoded[:, 1]
    X = StandardScaler().fit_transform(X)

    # Grid search random forest:
    param_grid = {'min_samples_split': [2, 10, 20],
                  'max_depth': [None, 2, 5, 10],
                  'min_samples_leaf': [1, 5, 10],
                  'max_leaf_nodes': [10, 20, 50, 100]
                  }

    do_grid_search(RandomForestRegressor(), param_grid, X, y)

    param_grid = {'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                  'degree': range(1, 4),
                  'epsilon': [1 * 10**n for n in range(-1, 1)],
                  'gamma': ['auto'] + [1 * 10**n for n in range(-1, 1)],
                  'coef0': [1 * 10**n for n in range(-4, 1)],
                  'tol': [1 * 10**n for n in range(-4, 1)],
                  'C': [1 * 10**n for n in range(-1, 1)]
                  }

    do_grid_search(SVR(kernel='poly'), param_grid, X, y)


if __name__ == '__main__':
    main(sys.argv[1:])
