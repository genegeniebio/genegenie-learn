'''
(c) University of Liverpool 2019

All rights reserved.

@author: neilswainston
'''
# pylint: disable=too-many-arguments
from sklearn import model_selection

from liv_learn.keras import Classifier


def k_fold_cross_valid(x_data, y_data,
                       tests=50,
                       test_size=0.05):
    '''k-fold cross validation.'''
    results = []

    for _ in range(tests):
        _k_fold_cross_valid((x_data, y_data),
                            results,
                            test_size)

    return results


def _k_fold_cross_valid(x_data, y_data,
                        results,
                        test_size=0.05):
    '''k-fold cross validation.'''
    x_train, x_test, y_train, y_test = \
        model_selection.train_test_split(x_data, y_data,
                                         test_size=test_size)

    model = Classifier(x_train, y_train)

    model.train()

    for test, pred in zip(y_test, model.predict(x_test)):
        results.append((test, pred))
