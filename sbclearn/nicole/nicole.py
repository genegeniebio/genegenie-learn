'''
sbclearn (c) University of Manchester 2018

sbclearn is licensed under the MIT License.

To view a copy of this license, visit <http://opensource.org/licenses/MIT/>.

@author:  neilswainston
'''
# pylint: disable=invalid-name
# pylint: disable=too-many-arguments
import itertools
import os
import sys

from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.ensemble.forest import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, train_test_split, \
    GridSearchCV
from sklearn.preprocessing.data import StandardScaler
from sklearn.svm import SVR
from sklearn.tree.tree import DecisionTreeRegressor

import numpy as np
import pandas as pd
from sbclearn.utils import aligner, plot, transformer
from synbiochem.utils import xl_converter


# from sklearn.ensemble.forest import RandomForestRegressor
# from sklearn.linear_model import LinearRegression
# from sklearn.tree.tree import DecisionTreeRegressor
def get_data(xl_filename, sources=None):
    '''Get data.'''
    df = _get_raw_data(xl_filename)

    # Filter rows:
    if sources:
        df = df.loc[df['source'].isin(sources)]

    df = aligner.align(df)
    df.to_csv('aligned.csv')

    learn_df = df.loc[:, ['dif_align_seq', 'geraniol']]
    learn_df.columns = ['seq', 'activity']
    learn_df.to_csv('learn.csv')

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
    df = df[df['seq'].notnull()]
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


def do_grid_search(estimator, X, y, cv, param_grid=None, verbose=False):
    '''Perform grid search.'''
    if not param_grid:
        param_grid = {}

    grid_search = GridSearchCV(estimator,
                               param_grid,
                               scoring='neg_mean_squared_error',
                               cv=cv,
                               verbose=verbose)

    grid_search.fit(X, y)

    res = grid_search.cv_results_

    for mean, params in sorted(zip(res['mean_test_score'], res['params']),
                               reverse=True):
        print (np.sqrt(-mean), params)

    print


def _hi_level_investigation(data):
    '''Perform high-level investigation.'''
    transformers = [
        transformer.OneHotTransformer(nucl=False),
        transformer.AminoAcidTransformer()]

    estimators = [
        LinearRegression(),
        DecisionTreeRegressor(),
        RandomForestRegressor(),
        ExtraTreesRegressor(),
        GradientBoostingRegressor(),
        SVR(kernel='poly')
    ]

    cv = 10

    for trnsfrmr, estimator in itertools.product(transformers, estimators):
        encoded = trnsfrmr.transform(data)
        X, y = encoded[:, 2:], encoded[:, 1]
        X = StandardScaler().fit_transform(X)

        mean, std = cross_valid_score(estimator, X, y, cv=cv)

        print '\t'.join([trnsfrmr.__class__.__name__,
                         estimator.__class__.__name__,
                         str((mean, std))])

    print


def _grid_search_extra_trees(X, y, cv):
    '''Grid search with ExtraTreesRegressor.'''
    param_grid = {'min_samples_split': [2, 5, 10],
                  'max_depth': [None, 1, 2, 5],
                  'min_samples_leaf': [2, 5, 10],
                  'max_leaf_nodes': [None, 2, 5],
                  'n_estimators': [10, 20, 50]
                  }

    do_grid_search(ExtraTreesRegressor(), X, y, cv, param_grid)


def _grid_search_svr(X, y, cv):
    '''Grid search with SVR.'''
    param_grid = {'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                  'degree': range(1, 4),
                  'epsilon': [1 * 10**n for n in range(-1, 1)],
                  'gamma': ['auto'] + [1 * 10**n for n in range(-1, 1)],
                  'coef0': [1 * 10**n for n in range(-4, 1)],
                  'tol': [1 * 10**n for n in range(-4, 1)],
                  'C': [1 * 10**n for n in range(-1, 1)]
                  }

    do_grid_search(SVR(kernel='poly'), X, y, cv, param_grid)


def _predict(estimator, X, y, tests=25, test_size=0.05):
    '''Predict.'''
    y_tests = []
    y_preds = []

    for _ in range(0, tests):
        X_train, X_test, y_train, y_test = \
            train_test_split(X, y, test_size=test_size)
        estimator.fit(X_train, y_train)
        y_tests.extend(y_test)
        y_preds.extend(estimator.predict(X_test))

    plot(y_tests, y_preds)


def main(args):
    '''main method.'''
    data = get_data(args[0], args[1:] if len(args) > 1 else None)
    _hi_level_investigation(data)

    encoded = transformer.AminoAcidTransformer().transform(data)
    X, y = encoded[:, 2:], encoded[:, 1]
    X = StandardScaler().fit_transform(X)

    # _grid_search_extra_trees(X, y, cv)
    # _grid_search_svr(X, y, cv)

    _predict(GradientBoostingRegressor(), X, y)


if __name__ == '__main__':
    main(sys.argv[1:])
