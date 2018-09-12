'''
sbclearn (c) University of Manchester 2018

sbclearn is licensed under the MIT License.

To view a copy of this license, visit <http://opensource.org/licenses/MIT/>.

@author:  neilswainston
'''
# pylint: disable=invalid-name
# pylint: disable=too-many-arguments
import itertools
import sys

from keras.optimizers import Adam

from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.ensemble.forest import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, train_test_split, \
    GridSearchCV
from sklearn.preprocessing.data import StandardScaler
from sklearn.svm import SVR
from sklearn.tree.tree import DecisionTreeRegressor

from gg_learn.keras import regress_lstm
from gg_learn.nicole import get_aligned_data, get_data
from gg_learn.utils import transformer
from gg_learn.utils.biochem_utils import get_ordinal_seq
import numpy as np


# from sklearn.ensemble.forest import RandomForestRegressor
# from sklearn.linear_model import LinearRegression
# from sklearn.tree.tree import DecisionTreeRegressor
def analyse(df):
    '''Analyse.'''
    X = get_ordinal_seq(df['seq'])
    y = df['geraniol'].fillna(0)

    score = regress_lstm(X, y, optimizer=Adam(lr=0.00025), batch_size=10, epochs=50)
    print('Score: %.2f RMSE' % (score))


def analyse_aligned(df):
    '''Analyse aligned data.'''
    aligned_data = get_aligned_data(df)

    _hi_level_investigation(aligned_data)

    encoded = transformer.AminoAcidTransformer().transform(aligned_data)
    X, y = encoded[:, 2:], encoded[:, 1]
    X = StandardScaler().fit_transform(X)

    # _grid_search_extra_trees(X, y, cv)
    # _grid_search_svr(X, y, cv)

    _grid_search_random_forest(X, y, cv=10)

    _predict(RandomForestRegressor(), X, y)


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

        scores = cross_val_score(estimator,
                                 X, y,
                                 scoring='neg_mean_squared_error',
                                 cv=cv,
                                 verbose=False)
        scores = np.sqrt(-scores)

        print '\t'.join([trnsfrmr.__class__.__name__,
                         estimator.__class__.__name__,
                         str((scores.mean(), scores.std()))])

    print


def _grid_search_random_forest(X, y, cv):
    '''Grid search with ExtraTreesRegressor.'''
    param_grid = {  # 'min_samples_split': [2, 5, 10],
        'max_depth': [None, 1, 2, 5],
        # 'min_samples_leaf': [2, 5, 10],
        'max_leaf_nodes': [None, 2, 5],
        'n_estimators': [10, 20, 50],
        # 'min_weight_fraction_leaf': [0, 0.1, 0.2]
    }

    do_grid_search(RandomForestRegressor(), X, y, cv, param_grid)


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


def main(args):
    '''main method.'''
    df = get_data(args[0], args[1:] if len(args) > 1 else None)
    df.to_csv('geraniol.csv')
    analyse(df)
    # analyse_aligned(df)


if __name__ == '__main__':
    main(sys.argv[1:])
